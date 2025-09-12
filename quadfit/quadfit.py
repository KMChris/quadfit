from __future__ import annotations
import numpy as np
from warnings import warn
from typing import Literal

from quadfitmodule import best_iou_quadrilateral as _best_iou_quad  # C accelerated
from quadfitmodule import finetune_quadrilateral as _finetune_quad  # C accelerated
from quadfitmodule import expand_quadrilateral as _expand_quad  # C accelerated
from quadfitmodule import simplify_polygon_dp as _simplify_dp  # C accelerated
from quadfitmodule import convex_hull_monotone as _convex_hull  # C accelerated



class QuadrilateralFitter:
    def __init__(self, polygon: np.ndarray | tuple | list | object):
        """
        Initialize a QuadrilateralFitter from input coordinates.

        Accepted input forms for `polygon`:
        - numpy.ndarray of shape (N, 2)
        - list/tuple of (x, y) pairs
        - any object exposing `.exterior.coords` (e.g., shapely.geometry.Polygon)
        - any object exposing `.coords` (e.g., shapely.geometry.LineString)
        """
        # Normalize input to numpy array of shape (N, 2)
        coords: np.ndarray | None = None
        # Duck-typing for shapely Polygon
        if hasattr(polygon, "exterior") and hasattr(getattr(polygon, "exterior"), "coords"):
            coords = np.array(polygon.exterior.coords, dtype=np.float64)
        # Duck-typing for LineString-like objects
        elif hasattr(polygon, "coords"):
            line_coords = np.array(polygon.coords, dtype=np.float64)
            warn("Input appears to be a LineString-like geometry. Results may be inaccurate; using a padded rectangle around it.")
            # Define a rectangle around the line with a margin
            min_x, min_y = np.min(line_coords, axis=0)
            max_x, max_y = np.max(line_coords, axis=0)
            margin = 10.0
            coords = np.array([
                [min_x - margin, min_y - margin],
                [min_x - margin, max_y + margin],
                [max_x + margin, max_y + margin],
                [max_x + margin, min_y - margin],
                [min_x - margin, min_y - margin],
            ], dtype=np.float64)
        elif isinstance(polygon, np.ndarray):
            if polygon.ndim != 2 or polygon.shape[1] != 2:
                raise AssertionError(f"Input polygon must have shape (N, 2). Got {polygon.shape}")
            coords = np.asarray(polygon, dtype=np.float64)
        elif isinstance(polygon, (list, tuple)):
            # Expect iterable of (x, y)
            if not all(isinstance(coord, (list, tuple)) and len(coord) == 2 for coord in polygon):
                raise AssertionError("Expected list/tuple of (x, y) coordinate pairs")
            coords = np.array(polygon, dtype=np.float64)
        else:
            raise TypeError(
                f"Unexpected input type: {type(polygon)}. Accepted: np.ndarray, list/tuple of (x,y), "
                f"or any object exposing .exterior.coords / .coords"
            )

        self._polygon_coords = coords

        # Compute convex hull once in C for hot-path
        self._hull_coords = _convex_hull(self._polygon_coords)
        # Internal state for staged results
        self._initial_quadrilateral: tuple | None = None
        self._refined_quadrilateral: tuple | None = None
        self._final_quadrilateral: tuple | None = None
        self._line_equations = None
        self._expanded_line_equations = None

    def fit(self, simplify_polygons_larger_than: int|None = 10, start_simplification_epsilon: float = 0.1,
        max_simplification_epsilon: float = 0.5, simplification_epsilon_increment: float = 0.02,
        max_initial_combinations: int = 300, random_seed: int | None = None,
        until: Literal["initial", "refined", "final"] = "final") -> tuple:
        """
        Fit a quadrilateral around the input polygon/point-cloud.

        The algorithm:
        1) Computes the convex hull (in C) of the input points.
        2) Finds an initial quadrilateral that maximizes IoU with the convex hull (sampling or full search).
        3) Refines the quadrilateral by reassigning points to sides and fitting TLS lines (in C).
        4) Expands the lines outward to guarantee the quadrilateral contains the convex hull.

        Parameters:
        - simplify_polygons_larger_than: If specified, perform a preliminary Douglas–Peucker simplification of the hull
          when it has more than this many vertices (speeds up initial search, may slightly reduce optimality).
        - start_simplification_epsilon / max_simplification_epsilon / simplification_epsilon_increment:
          Epsilon schedule for the Douglas–Peucker simplification.
        - max_initial_combinations: Cap on the number of candidate quadrilaterals (combinations C(N,4)) to test
          during the initial search. If 0 or >= C(N,4), a full search is used; otherwise random unique samples are used.
        - random_seed: RNG seed for deterministic sampling when the search is capped.

        Returns:
        - A tuple of four (x, y) points (clockwise) for the requested stage specified by `until`.
        """
        if until not in ("initial", "refined", "final"):
            raise ValueError("until must be one of: 'initial', 'refined', 'final'")
        self._initial_quadrilateral = self.__find_initial_quadrilateral(
            max_sides_to_simplify=simplify_polygons_larger_than,
            start_simplification_epsilon=start_simplification_epsilon,
            max_simplification_epsilon=max_simplification_epsilon,
            simplification_epsilon_increment=simplification_epsilon_increment,
            max_combinations=max_initial_combinations,
            random_seed=random_seed,
        )
        if until == "initial":
            # Skip later stages, return initial only
            self._refined_quadrilateral = None
            self._final_quadrilateral = None
            return self._initial_quadrilateral

        self._refined_quadrilateral = self.__finetune_guess()
        if until == "refined":
            self._final_quadrilateral = None
            return self._refined_quadrilateral

        self._final_quadrilateral = self.__expand_quadrilateral()
        return self._final_quadrilateral


    def __find_initial_quadrilateral(self, max_sides_to_simplify: int | None = 10,
                                     start_simplification_epsilon: float = 0.1,
                                     max_simplification_epsilon: float = 0.5,
                                     simplification_epsilon_increment: float = 0.02,
                                     max_combinations: int = 300,
                                     random_seed: int | None = None) -> tuple:
        """
        Compute the initial quadrilateral from convex-hull vertices.

        Chooses the 4-vertex combination with the highest IoU against the hull (full search or random sampling).

        Parameters mirror `fit()` for the simplification and search cap.

        Returns: tuple of four (x, y) points (clockwise) for the initial quadrilateral.
        """
        if max_sides_to_simplify is None:
            hull_coords = self._hull_coords
        else:
            hull_coords = self._hull_coords
            simp = _simplify_dp(
                hull_coords,
                max_sides_to_simplify,
                start_simplification_epsilon,
                max_simplification_epsilon,
                simplification_epsilon_increment,
                0.8,
            )
            hull_coords = np.asarray(simp, dtype=np.float64)

        quad_vertices = _best_iou_quad(hull_coords, max_combinations, random_seed)
        return tuple(map(tuple, quad_vertices))

    def __finetune_guess(self) -> tuple:
        """
        Finetune the quadrilateral by reassigning points to sides and fitting TLS lines.

        Returns: tuple of four (x, y) points for the refined quadrilateral.
        """

        # Use C accelerated finetuning: assign points to nearest side and fit TLS lines
        points = np.asarray(self._polygon_coords, dtype=np.float64)
        initv = np.array(self._initial_quadrilateral, dtype=np.float64)
        lines_obj, new_vertices = _finetune_quad(points, initv)
        # Store lines as Line objects
        self._line_equations = tuple(lines_obj)
        refined = tuple(map(tuple, new_vertices))
        self._refined_quadrilateral = refined
        return refined

    def __expand_quadrilateral(self) -> tuple:
        """
        Expand fitted lines outward to ensure the quadrilateral contains all hull points.

        Returns: tuple of four (x, y) points for the expanded quadrilateral.
        """
        # Delegate to accelerated C expansion using hull points
        hull_coords = self._hull_coords
        lines_obj, vertices = _expand_quad(self._line_equations, hull_coords)
        self._expanded_line_equations = tuple(lines_obj)
        quad = tuple(map(tuple, vertices))
        self._final_quadrilateral = quad
        return quad

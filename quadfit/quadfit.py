from __future__ import annotations
import numpy as np
from warnings import warn

from quadfitmodule import best_iou_quadrilateral as _best_iou_quad  # C accelerated
from quadfitmodule import finetune_quadrilateral as _finetune_quad  # C accelerated
from quadfitmodule import expand_quadrilateral as _expand_quad  # C accelerated
from quadfitmodule import simplify_polygon_dp as _simplify_dp  # C accelerated
from quadfitmodule import convex_hull_monotone as _convex_hull  # C accelerated

class QuadrilateralFitter:
    def __init__(self, polygon: np.ndarray | tuple | list | object):
        """
        Constructor for initializing the QuadrilateralFitter object.

        :param polygon: Coordinates of the input geometry. Accepted forms:
                         - numpy.ndarray of shape (N, 2)
                         - list/tuple of (x, y) pairs
                         - any object with .exterior.coords (e.g., shapely.geometry.Polygon)
                         - any object with .coords (e.g., shapely.geometry.LineString)
        """
        # Normalize input to numpy array of shape (N, 2)
        coords: np.ndarray | None = None
        # Duck-typing for shapely Polygon
        if hasattr(polygon, "exterior") and hasattr(getattr(polygon, "exterior"), "coords"):
            coords = np.array(polygon.exterior.coords, dtype=np.float64)
        # Duck-typing for shapely LineString or similar
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

        self._initial_guess = None

        self._line_equations = None
        self.fitted_quadrilateral = None
        self._expanded_line_equations = None
        self.expanded_quadrilateral = None

    def fit(self, simplify_polygons_larger_than: int|None = 10, start_simplification_epsilon: float = 0.1,
        max_simplification_epsilon: float = 0.5, simplification_epsilon_increment: float = 0.02,
        max_initial_combinations: int = 300, random_seed: int | None = None) -> \
            tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        Fits an irregular quadrilateral around the input polygon. The quadrilateral is optimized to minimize
        the Intersection over Union (IoU) with the input polygon.

        This method performs the following steps:
        1. Computes the convex hull of the input polygon.
        2. Finds an initial quadrilateral that closely approximates the convex hull.
        3. Refines this initial quadrilateral to ensure it fully circumscribes the convex hull.

        Note: The input polygon should be of shape (N, 2), where N is the number of vertices.

        :param simplify_polygons_larger_than: int | None. If a number is specified, the method will make a
                        preliminar Douglas-Peucker simplification of the Convex Hull if it has more than
                        simplify_polygons_larger_than vertices. This will speed up the process, but may
                        lead to a sub-optimal quadrilateral approximation.
        :param start_simplification_epsilon: float. The initial simplification epsilon to use if
                        simplify_polygons_larger_than is not None (for Douglas-Peucker simplification).
        :param max_simplification_epsilon: float. The maximum simplification epsilon to use if
                        simplify_polygons_larger_than is not None (for Douglas-Peucker simplification).
    :param simplification_epsilon_increment: float. The increment in the simplification epsilon to use if
                        simplify_polygons_larger_than is not None (for Douglas-Peucker simplification).
    :param max_initial_combinations: int. Limit na liczbę kombinacji kandydatów (C(N,4)) do sprawdzenia
            przy wyborze wstępnego czworokąta. Gdy 0 lub wartość >= C(N,4), wykonywane jest pełne
            przeszukanie; w przeciwnym razie losowo próbkowane jest max_initial_combinations unikatowych
            kombinacji.
    :param random_seed: int | None. Ziarno RNG dla losowego próbkowania kombinacji (deterministyczność wyników).

        :return: A tuple containing four tuples, each of which has two float elements representing the (x, y)
                coordinates of the quadrilateral's vertices. The vertices are order clockwise.

        :raises AssertionError: If the input polygon does not have a shape of (N, 2).
        """
        self._initial_guess = self.__find_initial_quadrilateral(
            max_sides_to_simplify=simplify_polygons_larger_than,
            start_simplification_epsilon=start_simplification_epsilon,
            max_simplification_epsilon=max_simplification_epsilon,
            simplification_epsilon_increment=simplification_epsilon_increment,
            max_combinations=max_initial_combinations,
            random_seed=random_seed,
        )
        self.fitted_quadrilateral = self.__finetune_guess()
        self.expanded_quadrilateral = self.__expand_quadrilateral()
        return self.fitted_quadrilateral


    def __find_initial_quadrilateral(self, max_sides_to_simplify: int | None = 10,
                                     start_simplification_epsilon: float = 0.1,
                                     max_simplification_epsilon: float = 0.5,
                                     simplification_epsilon_increment: float = 0.02,
                                     max_combinations: int = 300,
                                     random_seed: int | None = None) -> tuple:
        """
        Internal method to find the initial approximating quadrilateral based on the vertices of the Convex Hull.
        To find the initial quadrilateral, we iterate through all 4-vertex combinations of the Convex Hull vertices
        and find the one with the highest Intersection over Union (IoU) with the Convex Hull. It will ensure that
        it is the best possible quadrilateral approximation to the input polygon.
        :param max_sides_to_simplify: int|None. If a number is specified, the method will make a
                        preliminar Douglas-Peucker simplification of the Convex Hull if it has more than
                        max_sides_to_simplify vertices. This will speed up the process, but may
                        lead to a sub-optimal quadrilateral approximation.
        :param start_simplification_epsilon: float. The initial simplification epsilon to use if
                        max_sides_to_simplify is not None (for Douglas-Peucker simplification).
        :param max_simplification_epsilon: float. The maximum simplification epsilon to use if
                        max_sides_to_simplify is not None (for Douglas-Peucker simplification).
        :param simplification_epsilon_increment: float. The increment in the simplification epsilon to use if
                        max_sides_to_simplify is not None (for Douglas-Peucker simplification).

        :param max_combinations: int. The maximum number of combinations to try. If the number of combinations
                is larger than this number, the method will only run random max_combinations combinations.
        :param random_seed: int | None. RNG seed for deterministic random sampling when max_combinations < C(N,4).

        :return: tuple: Cztery wierzchołki (x, y) początkowego czworokąta.
        """
        # Przygotuj wierzchołki otoczki wypukłej (opcjonalnie uproszczone) jako tablicę numpy.
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

        # C zwraca 4 wierzchołki (np.ndarray shape (4,2)); zwracamy jako tuple of tuples
        quad_vertices = _best_iou_quad(hull_coords, max_combinations, random_seed)
        return tuple(map(tuple, quad_vertices))

    def __finetune_guess(self) -> tuple:
        """
        Internal method to finetune the initial quadrilateral approximation to adjust to the input polygon.
        The method works by deciding which point of the initial polygon belongs to which side of the input polygon
        and fitting a line to each side of the input polygon. The intersection points between the lines will
        be the vertices of the new quadrilateral.

    :return: tuple. Cztery wierzchołki (x, y) dopracowanego czworokąta.
        """

        # Use C accelerated finetuning: assign points to nearest side and fit TLS lines
        points = np.asarray(self._polygon_coords, dtype=np.float64)
        initv = np.array(self._initial_guess, dtype=np.float64)
        lines_obj, new_vertices = _finetune_quad(points, initv)
        # Store lines as Line objects
        self._line_equations = tuple(lines_obj)
        return tuple(map(tuple, new_vertices))

    def __expand_quadrilateral(self) -> tuple:
        """
        Internal method that expands the initial quadrilateral approximation to make sure it contains all the vertices
        of the input polygon Convex Hull.
        Method:
            1. Move each line in their orthogonal direction (outwards) until it contains (or intersects)
               all the points of the Convex Hull in its inward direction
            2. Find the intersection points between the lines to calculate the vertices of the
               new expanded quadrilateral

        :param quadrilateral: tuple. Cztery wierzchołki początkowego czworokąta.

    :return: tuple. Cztery wierzchołki (x, y) rozszerzonego czworokąta.
        """
        # Delegate to accelerated C expansion using hull points
        hull_coords = self._hull_coords
        lines_obj, vertices = _expand_quad(self._line_equations, hull_coords)
        self._expanded_line_equations = tuple(lines_obj)
        quad = tuple(map(tuple, vertices))
        self.expanded_quadrilateral = quad
        return quad

    # Backward-compat alias used by some examples/tests
    @property
    def tight_quadrilateral(self):
        return getattr(self, "expanded_quadrilateral", None)


    # -------------------------------- HELPER METHODS -------------------------------- #

    # Removed Shapely-based IoU to avoid hard dependency.

    # (usunięto nieużywaną metodę __sign)

    def plot(self):
        """
        Plot the convex hull and the best-fitting quadrilateral for debugging purposes.
        This function imports matplotlib.pyplot locally, so the library is not required for the entire class.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("This function requires matplotlib to be installed. Please install it first.")

        # Plot the original polygon as a set of alpha 0.4 points
        plt.plot(self._polygon_coords[:, 0], self._polygon_coords[:, 1], alpha=0.3,  linestyle='-', marker='o', label='Input Polygon')

        # Plot the convex hull as a filled polygon using cached hull coords
        hx, hy = self._hull_coords[:, 0], self._hull_coords[:, 1]
        plt.fill(hx, hy, alpha=0.4, label='Convex Hull', color='orange')

        # Plot the initial quadrilateral if it exists as a semi-transparent dashed line
        if self._initial_guess is not None:
            ix, iy = zip(*self._initial_guess)
            plt.plot(ix + (ix[0],), iy + (iy[0],), linestyle='--', alpha=0.5, color='green', label='Initial Guess')

        # Plot the best quadrilateral if it exists
        if self.fitted_quadrilateral is not None:
            x, y = zip(*self.fitted_quadrilateral)
            plt.plot(x + (x[0],), y + (y[0],), label='Fitted Quadrilateral')
            plt.scatter(x, y, marker='x', color='red')

        plt.axis('equal')
        plt.xlabel('X')
        # Reverse Y axis
        plt.ylabel('Y')
        plt.title('Quadrilateral Fitting')
        plt.legend()
        ax = plt.gca()  # Get current axes
        ax.invert_yaxis()
        plt.grid(True)
        plt.show()



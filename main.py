from quadfit import QuadrilateralFitter
import numpy as np
import cv2
from matplotlib import pyplot as plt
import timeit
from typing import Iterable, Tuple

def yugioh_test():
    image = cv2.cvtColor(cv2.imread('./resources/input_sample.jpg'), cv2.COLOR_BGR2RGB)
    true_corners = np.array([[50., 100.], [370., 0.], [421., 550.], [0., 614.], [50., 100.]], dtype=np.float32)

    # Generate the noisy corners
    sides = [np.linspace([x1, y1], [x2, y2], 25) + np.random.normal(scale=5, size=(25, 2))
             for (x1, y1), (x2, y2) in zip(true_corners[:-1], true_corners[1:])]
    noisy_corners = np.concatenate(sides, axis=0)

    # To simplify, we will clip the corners to be within the image
    noisy_corners[:, 0] = np.clip(noisy_corners[:, 0], a_min=0., a_max=image.shape[1])
    noisy_corners[:, 1] = np.clip(noisy_corners[:, 1], a_min=0., a_max=image.shape[0])

    start = timeit.default_timer()
    fitter = QuadrilateralFitter(polygon=noisy_corners)
    after_fitter = timeit.default_timer()
    # If only the initial guess is needed, skip later stages for speed
    result = fitter.fit(simplify_polygons_larger_than=30, until="initial")
    after_fit = timeit.default_timer()
    tight_quadrilateral = result
    after_tight = timeit.default_timer()

    print(f"QuadrilateralFitter init: {after_fitter - start:.6f} seconds")
    print(f"fit() call: {after_fit - after_fitter:.6f} seconds")
    print(f"tight_quadrilateral access: {after_tight - after_fit:.6f} seconds")
    print(f"Total: {after_tight - start:.6f} seconds")

def plot_fitter(fitter: QuadrilateralFitter):
    """Plot the input points, convex hull, and any available quadrilaterals from the fitter."""
    # Input polygon / points
    pc = fitter._polygon_coords  # internal, but OK for debugging/demo
    plt.plot(pc[:, 0], pc[:, 1], alpha=0.3, linestyle='-', marker='o', label='Input Polygon')

    # Convex hull
    hx, hy = fitter._hull_coords[:, 0], fitter._hull_coords[:, 1]
    plt.fill(hx, hy, alpha=0.4, label='Convex Hull', color='orange')

    # Initial quadrilateral if present
    if fitter._initial_quadrilateral is not None:
        ix, iy = zip(*fitter._initial_quadrilateral)
        plt.plot(ix + (ix[0],), iy + (iy[0],), linestyle='--', alpha=0.5, color='green', label='Initial Guess')

    # Refined quadrilateral (after TLS) if present
    if getattr(fitter, "_refined_quadrilateral", None) is not None:
        rx, ry = zip(*fitter._refined_quadrilateral)
        plt.plot(rx + (rx[0],), ry + (ry[0],), linestyle='-.', alpha=0.7, color='blue', label='Refined (TLS)')

    # Final quadrilateral if present
    if fitter._final_quadrilateral is not None:
        x, y = zip(*fitter._final_quadrilateral)
        plt.plot(x + (x[0],), y + (y[0],), label='Final Quadrilateral')
        plt.scatter(x, y, marker='x', color='red')

    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    # IoU metrics if available
    title = 'Quadrilateral Fitting'
    try:
        if fitter._initial_quadrilateral is not None:
            iou_init = fitter.iou_vs_hull(fitter._initial_quadrilateral)
            title += f" | IoU init={iou_init:.3f}"
        if getattr(fitter, "_refined_quadrilateral", None) is not None:
            iou_ref = fitter.iou_vs_hull(fitter._refined_quadrilateral)
            title += f", refined={iou_ref:.3f}"
        if fitter._final_quadrilateral is not None:
            iou_fin = fitter.iou_vs_hull(fitter._final_quadrilateral)
            title += f", final={iou_fin:.3f}"
    except Exception:
        pass
    plt.title(title)
    plt.legend()
    ax = plt.gca()
    ax.invert_yaxis()
    plt.grid(True)
    plt.show()

def benchmark_case(name: str, data: np.ndarray, repeats: int = 3, simplify_polygons_larger_than: int = 30):
    def time_once(until: str) -> Tuple[float, float, float]:
        t0 = timeit.default_timer()
        fitter = QuadrilateralFitter(polygon=data)
        t1 = timeit.default_timer()
        _ = fitter.fit(simplify_polygons_larger_than=simplify_polygons_larger_than, until=until)
        t2 = timeit.default_timer()
        return (t1 - t0, t2 - t1, t2 - t0)

    results = {k: [] for k in ("initial", "refined", "final")}
    for k in results.keys():
        for _ in range(repeats):
            results[k].append(time_once(k))

    def avg(triples: Iterable[Tuple[float, float, float]]):
        arr = np.array(triples, dtype=float)
        return arr.mean(axis=0)

    ai, ar, af = avg(results["initial"]), avg(results["refined"]), avg(results["final"])
    # Compute IoU for each stage once (not averaged) for a representative run
    f = QuadrilateralFitter(polygon=data)
    q_init = f.fit(simplify_polygons_larger_than=simplify_polygons_larger_than, until="initial")
    iou_init = f.iou_vs_hull(q_init)
    q_ref = f.fit(simplify_polygons_larger_than=simplify_polygons_larger_than, until="refined")
    iou_ref = f.iou_vs_hull(q_ref)
    q_fin = f.fit(simplify_polygons_larger_than=simplify_polygons_larger_than, until="final")
    iou_fin = f.iou_vs_hull(q_fin)

    print(f"\nBenchmark: {name} (N={len(data)})  repeats={repeats}")
    print(f"  until=initial -> init: {ai[0]:.6f}s, fit: {ai[1]:.6f}s, total: {ai[2]:.6f}s | IoU={iou_init:.3f}")
    print(f"  until=refined -> init: {ar[0]:.6f}s, fit: {ar[1]:.6f}s, total: {ar[2]:.6f}s | IoU={iou_ref:.3f}")
    print(f"  until=final   -> init: {af[0]:.6f}s, fit: {af[1]:.6f}s, total: {af[2]:.6f}s | IoU={iou_fin:.3f}")

if __name__ == '__main__':
    # 1. Deformed trapezoid
    num_points = 20
    left_side = np.linspace([0.2, 0.3], [0.25, 0.8], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    top_side = np.linspace([0.25, 0.8], [0.8, 0.7], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    right_side = np.linspace([0.8, 0.7], [0.8, 0.3], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    bottom_side = np.linspace([0.8, 0.3], [0.2, 0.3], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    deformed_trapezoid = np.vstack([top_side, right_side, bottom_side, left_side])

    deformed_trapezoid = np.array([[433.0, 144.0], [432.0, 145.0], [417.0, 145.0], [416.0, 146.0], [391.0, 146.0], [390.0, 147.0], [367.0, 147.0], [366.0, 148.0], [320.0, 148.0], [319.0, 149.0], [301.0, 149.0], [300.0, 150.0], [293.0, 150.0], [292.0, 151.0], [281.0, 151.0], [280.0, 152.0], [249.0, 152.0], [248.0, 153.0], [237.0, 153.0], [236.0, 154.0], [198.0, 154.0], [197.0, 155.0], [181.0, 155.0], [180.0, 156.0], [148.0, 156.0], [147.0, 155.0], [146.0, 155.0], [144.0, 157.0], [144.0, 181.0], [145.0, 182.0], [145.0, 187.0], [146.0, 188.0], [146.0, 189.0], [147.0, 190.0], [147.0, 194.0], [148.0, 195.0], [148.0, 207.0], [149.0, 208.0], [149.0, 213.0], [150.0, 214.0], [150.0, 218.0], [151.0, 219.0], [151.0, 224.0], [152.0, 225.0], [152.0, 246.0], [153.0, 247.0], [153.0, 251.0], [154.0, 252.0], [154.0, 253.0], [155.0, 254.0], [155.0, 259.0], [156.0, 260.0], [156.0, 273.0], [157.0, 274.0], [157.0, 280.0], [158.0, 281.0], [158.0, 284.0], [159.0, 285.0], [159.0, 294.0], [160.0, 295.0], [160.0, 313.0], [161.0, 314.0], [161.0, 320.0], [162.0, 321.0], [162.0, 324.0], [163.0, 325.0], [163.0, 332.0], [164.0, 333.0], [164.0, 344.0], [165.0, 345.0], [165.0, 349.0], [166.0, 350.0], [166.0, 353.0], [167.0, 354.0], [167.0, 361.0], [168.0, 362.0], [168.0, 381.0], [169.0, 382.0], [169.0, 388.0], [170.0, 389.0], [170.0, 392.0], [171.0, 393.0], [171.0, 399.0], [172.0, 400.0], [172.0, 413.0], [173.0, 414.0], [173.0, 422.0], [174.0, 423.0], [174.0, 430.0], [175.0, 431.0], [175.0, 442.0], [176.0, 443.0], [176.0, 451.0], [177.0, 452.0], [177.0, 456.0], [178.0, 457.0], [178.0, 460.0], [179.0, 461.0], [179.0, 484.0], [178.0, 485.0], [178.0, 486.0], [177.0, 487.0], [176.0, 487.0], [175.0, 488.0], [174.0, 488.0], [171.0, 491.0], [171.0, 503.0], [172.0, 504.0], [172.0, 507.0], [173.0, 508.0], [173.0, 509.0], [174.0, 510.0], [182.0, 510.0], [183.0, 511.0], [200.0, 511.0], [201.0, 510.0], [211.0, 510.0], [212.0, 509.0], [228.0, 509.0], [229.0, 508.0], [245.0, 508.0], [246.0, 507.0], [281.0, 507.0], [282.0, 506.0], [290.0, 506.0], [291.0, 505.0], [296.0, 505.0], [297.0, 504.0], [310.0, 504.0], [311.0, 503.0], [333.0, 503.0], [334.0, 502.0], [338.0, 502.0], [339.0, 501.0], [340.0, 501.0], [341.0, 500.0], [343.0, 500.0], [344.0, 499.0], [345.0, 499.0], [346.0, 498.0], [347.0, 498.0], [349.0, 496.0], [350.0, 496.0], [351.0, 495.0], [358.0, 495.0], [359.0, 494.0], [370.0, 494.0], [371.0, 493.0], [373.0, 493.0], [374.0, 492.0], [382.0, 492.0], [383.0, 491.0], [399.0, 491.0], [400.0, 490.0], [409.0, 490.0], [410.0, 489.0], [416.0, 489.0], [417.0, 488.0], [423.0, 488.0], [424.0, 489.0], [433.0, 489.0], [434.0, 490.0], [438.0, 490.0], [439.0, 491.0], [445.0, 491.0], [446.0, 492.0], [456.0, 492.0], [457.0, 493.0], [465.0, 493.0], [466.0, 494.0], [471.0, 494.0], [472.0, 495.0], [486.0, 495.0], [487.0, 496.0], [491.0, 496.0], [492.0, 495.0], [556.0, 495.0], [557.0, 496.0], [558.0, 495.0], [558.0, 492.0], [559.0, 491.0], [559.0, 486.0], [558.0, 485.0], [558.0, 479.0], [557.0, 478.0], [557.0, 474.0], [556.0, 473.0], [556.0, 468.0], [555.0, 467.0], [555.0, 459.0], [554.0, 458.0], [554.0, 453.0], [553.0, 452.0], [553.0, 449.0], [552.0, 448.0], [552.0, 440.0], [551.0, 439.0], [551.0, 426.0], [550.0, 425.0], [550.0, 416.0], [549.0, 415.0], [549.0, 408.0], [548.0, 407.0], [548.0, 399.0], [547.0, 398.0], [547.0, 389.0], [546.0, 388.0], [546.0, 385.0], [545.0, 384.0], [545.0, 382.0], [544.0, 381.0], [544.0, 378.0], [543.0, 377.0], [543.0, 371.0], [542.0, 370.0], [542.0, 355.0], [543.0, 354.0], [543.0, 349.0], [542.0, 348.0], [542.0, 340.0], [541.0, 339.0], [541.0, 337.0], [540.0, 336.0], [540.0, 333.0], [539.0, 332.0], [539.0, 330.0], [538.0, 329.0], [538.0, 327.0], [537.0, 326.0], [537.0, 323.0], [536.0, 322.0], [536.0, 295.0], [535.0, 294.0], [535.0, 277.0], [534.0, 276.0], [534.0, 269.0], [533.0, 268.0], [533.0, 264.0], [532.0, 263.0], [532.0, 257.0], [531.0, 256.0], [531.0, 248.0], [530.0, 247.0], [530.0, 243.0], [529.0, 242.0], [529.0, 238.0], [528.0, 237.0], [528.0, 231.0], [527.0, 230.0], [527.0, 215.0], [526.0, 214.0], [526.0, 206.0], [525.0, 205.0], [525.0, 201.0], [524.0, 200.0], [524.0, 195.0], [523.0, 194.0], [523.0, 184.0], [522.0, 183.0], [522.0, 178.0], [521.0, 177.0], [521.0, 175.0], [520.0, 174.0], [520.0, 167.0], [519.0, 166.0], [519.0, 152.0], [518.0, 151.0], [518.0, 146.0], [517.0, 145.0], [516.0, 145.0], [515.0, 144.0]])

    # No noise deformed trapezoid
    # 1. Deformed trapezoid
    num_points = 20
    left_side = np.linspace([0.2, 0.3], [0.25, 0.8], num_points)
    top_side = np.linspace([0.25, 0.8], [0.8, 0.7], num_points)
    right_side = np.linspace([0.8, 0.7], [0.8, 0.3], num_points)
    bottom_side = np.linspace([0.8, 0.3], [0.2, 0.3], num_points)
    no_noise_deformed_trapezoid = np.vstack([top_side, right_side, bottom_side, left_side])

    # 2. Perfect square
    square = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])

    # 3. Deformed circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 + 0.4 * np.cos(theta) + np.random.normal(scale=0.03, size=theta.shape)
    circle_y = 0.5 + 0.4 * np.sin(theta) + np.random.normal(scale=0.03, size=theta.shape)
    deformed_circle = np.vstack((circle_x, circle_y)).T

    # Deformed trapezoid that looks almost like a triangle
    num_points = 20
    left_side = np.linspace([0.2, 0.2], [0.2, 0.8], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    top_side = np.linspace([0.2, 0.8], [0.5, 0.8], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    right_side = np.linspace([0.5, 0.8], [0.75, 0.2], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    bottom_side = np.linspace([0.75, 0.2], [0.2, 0.2], num_points) + np.random.normal(scale=0.01, size=(num_points, 2))
    almost_triangle_trapezoid = np.vstack([top_side, right_side, bottom_side, left_side])

    # Running the tests (benchmark + plots)
    test_data = [
        ("deformed_trapezoid", deformed_trapezoid),
        ("square", square),
        ("deformed_circle", deformed_circle),
        ("almost_triangle_trapezoid", almost_triangle_trapezoid),
    ]

    # Benchmarks: compare until=initial/refined/final timings
    for name, data in test_data:
        benchmark_case(name, np.asarray(data, dtype=float), repeats=10)

    for _, data in test_data:
        fitter = QuadrilateralFitter(polygon=np.asarray(data, dtype=float))
        quad = fitter.fit()
        plot_fitter(fitter)

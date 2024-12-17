import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os


# Функция для проверки, находится ли точка внутри полуплоскости, определённой ребром
def is_inside(p, v1, v2):
    return (v2[0] - v1[0]) * (p[1] - v1[1]) > (v2[1] - v1[1]) * (p[0] - v1[0])


# Функция для нахождения точки пересечения двух отрезков
def find_intersection(v1, v2, start, end):
    delta1 = (v1[0] - v2[0], v1[1] - v2[1])
    delta2 = (start[0] - end[0], start[1] - end[1])
    n1 = v1[0] * v2[1] - v1[1] * v2[0]
    n2 = start[0] * end[1] - start[1] * end[0]
    denominator = 1.0 / (delta1[0] * delta2[1] - delta1[1] * delta2[0])
    return (n1 * delta2[0] - n2 * delta1[0]) * denominator, (n1 * delta2[1] - n2 * delta1[1]) * denominator


# Алгоритм отсечения многоугольников по методу Сазеранда-Ходжмана
def clip_polygon(subject_polygon, clip_polygon):
    result = subject_polygon
    prev_vertex = clip_polygon[-1]
    for curr_vertex in clip_polygon:
        temp_input = result
        result = []
        last_point = temp_input[-1]
        for point in temp_input:
            if is_inside(point, prev_vertex, curr_vertex):
                if not is_inside(last_point, prev_vertex, curr_vertex):
                    result.append(find_intersection(prev_vertex, curr_vertex, last_point, point))
                result.append(point)
            elif is_inside(last_point, prev_vertex, curr_vertex):
                result.append(find_intersection(prev_vertex, curr_vertex, last_point, point))
            last_point = point
        prev_vertex = curr_vertex
    return result


def liang_barsky_clipping(segments, clip_rectangle):
    output_segments = []
    
    if not clip_rectangle:
        return output_segments

    clip_points = clip_rectangle[0]
    x_min = min(p[0] for p in clip_points)
    y_min = min(p[1] for p in clip_points)
    x_max = max(p[0] for p in clip_points)
    y_max = max(p[1] for p in clip_points)

    for segment in segments:
        start, end = segment
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        p = [-dx, dx, -dy, dy]
        q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]

        u1, u2 = 0.0, 1.0
        is_accepted = True

        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    is_accepted = False
                    break
                continue
            t = qi / pi
            if pi < 0:
                if t > u2:
                    is_accepted = False
                    break
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    is_accepted = False
                    break
                if t < u2:
                    u2 = t

        if is_accepted and u1 <= u2:
            clipped_start = (x1 + u1 * dx, y1 + u1 * dy)
            clipped_end = (x1 + u2 * dx, y1 + u2 * dy)
            output_segments.append((clipped_start, clipped_end))

    return output_segments


# Функция для рисования многоугольников
def draw_polygon_plot(subject, clip, clipped):
    plt.figure()
    plt.fill(*zip(*subject), edgecolor='r', fill=False, linewidth=2, label='Subject Polygon')
    plt.scatter(*zip(*subject), color='r', zorder=5)
    plt.fill(*zip(*clip), edgecolor='b', fill=False, linewidth=2, label='Clip Polygon')
    plt.scatter(*zip(*clip), color='b', zorder=5)
    plt.fill(*zip(*clipped), edgecolor='g', fill=False, linewidth=2, label='Clipped Polygon')
    plt.scatter(*zip(*clipped), color='g', zorder=5)
    plt.legend()
    plt.title('Polygon Clipping (Sutherland-Hodgman)')

def plot_liang_barsky(segments, clipper, clipped_segments):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    for idx, seg in enumerate(segments):
        x_vals, y_vals = zip(*seg)
        plt.plot(x_vals, y_vals, color='blue', linewidth=2, label='Initial Segments' if idx == 0 else "")

    if clipper:
        clip_points = clipper[0]
        closed_clip = list(clip_points) + [clip_points[0]]
        x_clip, y_clip = zip(*closed_clip)
        plt.plot(x_clip, y_clip, color='red', linewidth=2, label='Clipper')

    for idx, seg in enumerate(clipped_segments):
        x_vals, y_vals = zip(*seg)
        plt.plot(x_vals, y_vals, color='green', linewidth=2, label='Clipped Segments' if idx == 0 else "")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title('Liang-Barsky Clipping')
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# Функция для генерации выпуклого многоугольника с помощью ConvexHull
def generate_convex(n):
    points = np.random.rand(n, 2) * 100
    hull = ConvexHull(points)
    return points[hull.vertices]


# Генерация многоугольника с случайными точками в пределах заданной области
def create_simple_polygon(n, bounding_box):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = np.random.rand(n) * (max(bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]) * 0.5)
    points = np.vstack((radius * np.cos(angle), radius * np.sin(angle))).T
    center = np.array([(bounding_box[0] + bounding_box[1]) / 2, (bounding_box[2] + bounding_box[3]) / 2])
    points += center
    return points


# Функция для ввода координат вручную
def manual_input(n_subject, n_clip):
    subject_polygon = []
    clip_polygon = []

    print(f"Введите {n_subject} пары координат для произвольного полигона:")
    for i in range(n_subject):
        x = float(input(f"Введите координату x{i + 1} для произвольного полигона: "))
        y = float(input(f"Введите координату y{i + 1} для произвольного полигона: "))
        subject_polygon.append((x, y))

    print(f"Введите {n_clip} пары координат для выпуклого отсекателя:")
    for i in range(n_clip):
        x = float(input(f"Введите координату x{i + 1} для выпуклого отсекателя: "))
        y = float(input(f"Введите координату y{i + 1} для выпуклого отсекателя: "))
        clip_polygon.append((x, y))

    return subject_polygon, clip_polygon

# Чтение данных из файла для алгоритма Лианга-Барски
def read_liang_barsky_data():
    filename = input("Введите имя файла с данными (если файл в текущей папке, просто укажите имя):\n")
    
    file_path = os.path.join(os.getcwd(), filename)

    try:
        with open(file_path, 'r') as f:
            n = int(f.readline())  # Считываем количество отрезков
            segments = []
            for _ in range(n):
                x1, y1, x2, y2 = map(float, f.readline().split())
                segments.append(((x1, y1), (x2, y2)))
            
            clipper = []
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, f.readline().split())
            clipper.append(((x1, y1), (x2, y2), (x3, y3), (x4, y4)))
        
        return segments, clipper
    
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        return [], []
    except ValueError:
        print("Ошибка: неверный формат данных в файле.")
        return [], []


# Основная функция
def main():
    n_clip = 5
    n_subject = 4

    while True:
        choose = input("1.Алгоритм Лианга-Барски\n2.Отсечение выпуклого многоугольника\n")
        if choose == "1":
            segments, clipper = read_liang_barsky_data()
            break
        elif choose == "2":
            while True:
                choose1 = input("\t1. Ручной ввод координат.\n\t2. Автоматическая генерация координат.\n")
                if choose1 == "1":
                    subject_polygon, clip_polygon = manual_input(n_subject, n_clip)
                    break
                elif choose1 == "2":
                    subject_polygon, clip_polygon = generate_polygons(n_subject, n_clip)
                    break
                else:
                    print("Неверный формат ввода\n")
            break
        else:
            print("Неверный формат ввода\n")


    if (choose == "1"):
        print("Liang-Barsky Segments:")
        for point in segments:
            print(f"({point[0]}, {point[1]})")

        print("Liang-Barsky Clipper")
        print(clipper)

        clipped_segments = liang_barsky_clipping(segments, clipper)

        print(clipped_segments)

        plot_liang_barsky(segments, clipper, clipped_segments)

    else:
        print("Clip Polygon Points:")
        for point in clip_polygon:
            print(f"({point[0]}, {point[1]})")

        print("Subject Polygon Points:")
        for point in subject_polygon:
            print(f"({point[0]}, {point[1]})")

        clipped_polygon = clip_polygon(subject_polygon, clip_polygon)

        print("Clipped Polygon Points:")
        for point in clipped_polygon:
            print(f"({point[0]}, {point[1]})")

        draw_polygon_plot(subject_polygon, clip_polygon, clipped_polygon)

if __name__ == "__main__":
    main()

import numpy as np
from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor, TransferFunction
from volume.volume import GradientVolume, Volume
from collections.abc import ValuesView
import math


# TODO: Implement trilinear interpolation
def trilinear_interpolation(volume: Volume, x: float, y: float, z: float):
    x0 = int(math.floor(x))
    x1 = x0 + 1
    y0 = int(math.floor(y))
    y1 = y0 + 1
    z0 = int(math.floor(z))
    z1 = z0 + 1
    if (x0 < 0 or y0 < 0 or z0 < 0 or x0 >= volume.dim_x or y0 >= volume.dim_y or z0 >= volume.dim_z or
            x1 < 0 or y1 < 0 or z1 < 0 or x1 >= volume.dim_x or y1 >= volume.dim_y or z1 >= volume.dim_z):
        return 0

    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    zd = (z - z0) / (z1 - z0)

    c000 = volume.data[x0, y0, z0]
    c100 = volume.data[x1, y0, z0]
    c001 = volume.data[x0, y0, z1]
    c101 = volume.data[x1, y0, z1]
    c010 = volume.data[x0, y1, z0]
    c110 = volume.data[x1, y1, z0]
    c011 = volume.data[x0, y1, z1]
    c111 = volume.data[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * (xd)
    c01 = c001 * (1 - xd) + c101 * (xd)
    c10 = c010 * (1 - xd) + c110 * (xd)
    c11 = c011 * (1 - xd) + c111 * (xd)

    c0 = c00 * (1 - yd) + c10 * (yd)
    c1 = c01 * (1 - yd) + c11 * (yd)

    c = c0 * (1 - zd) + c1 * (zd)

    return c


def get_voxel(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
        return 0

    x = int(math.floor(x))
    y = int(math.floor(y))
    z = int(math.floor(z))

    return volume.data[x, y, z]


class RaycastRendererImplementation(RaycastRenderer):
    """
    Class to be implemented.
    """

    def clear_image(self):
        """Clears the image data"""
        self.image.fill(0)

    def render_slicer(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2
        print(np.sum(image))
        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                # Get the voxel coordinate X
                voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                     volume_center[0]

                # Get the voxel coordinate Y
                voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                     volume_center[1]

                # Get the voxel coordinate Z
                voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                     volume_center[2]
                # print(voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                # Get voxel value
                value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)

                # Normalize value to be between 0 and 1
                red = value / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha
                # if red>127:
                # print(red, image[(j * image_size + i) * 4])
        if np.sum(image) > 0:
            print("End of slicing:", np.sum(image))
        return

    # TODO: Implement MIP function
    def render_mip(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # w vector. See documentation in parent's class
        w_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2
        print(np.sum(image))
        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 1 if self.interactive_mode else 1
        img_neg = -1 * image_center
        z_step = 5

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                max = 0
                for k in range(int(img_neg), int(image_center), z_step):

                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         w_vector[0] * (k) + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         w_vector[1] * (k) + volume_center[1]

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         w_vector[2] * (k) + volume_center[2]
                    # Get voxel value
                    value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                    voxel_coordinate_z)
                    if (value > max):
                        max = value
                    # if value<0: print("Inside:", max, value)
                # Normalize value to be between 0 and 1
                red = max / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha
            print("==============", i)

        # TODO: add colouring

    # TODO: Implement Compositing function. TFColor is already imported. self.tfunc is the current transfer function.
    def render_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # w vector. See documentation in parent's class
        w_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2
        print(np.sum(image))
        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 1 if self.interactive_mode else 1
        # img_neg = -1 * image_center

        self.tfunc.set_test_function()
        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                colorAll = TFColor(0, 0, 0, 1)
                for k in range(int(-image_size / 2), int(image_size / 2), 5):

                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         w_vector[0] * (k) + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         w_vector[1] * (k) + volume_center[1]

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         w_vector[2] * (k) + volume_center[2]
                    if (voxel_coordinate_x < 0 or voxel_coordinate_x >= volume.dim_x or \
                            voxel_coordinate_y < 0 or voxel_coordinate_y >= volume.dim_y or \
                            voxel_coordinate_z < 0 or voxel_coordinate_z >= volume.dim_z):
                        continue

                    value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                    voxel_coordinate_z)
                    if (value == 0):
                        continue
                    # if (value > max):
                    #   max = value

                    color = self.tfunc.get_color(value)

                    # Normalize value to be between 0 and 1
                    colorAll.r = color.a * color.r + (1 - color.a) * colorAll.r
                    colorAll.g = color.a * color.g + (1 - color.a) * colorAll.g
                    colorAll.b = color.a * color.b + (1 - color.a) * colorAll.b
                    # colorAll.a = color.a

                    # Compute the color value (0...255)
                    red = math.floor(colorAll.r * 255) if colorAll.r < 1.0 else 255
                    green = math.floor(colorAll.g * 255) if colorAll.g < 1.0 else 255
                    blue = math.floor(colorAll.b * 255) if colorAll.b < 1.0 else 255
                    alpha = math.floor(colorAll.a * 255) if colorAll.a < 1.0 else 255

                    # Assign color to the pixel i, j
                    image[(j * image_size + i) * 4] = red
                    image[(j * image_size + i) * 4 + 1] = green
                    image[(j * image_size + i) * 4 + 2] = blue
                    image[(j * image_size + i) * 4 + 3] = alpha
                    print("i j k:", i)
        # print("value", value, "max", max)
        print("DONE")

    # TODO: Implement Compositing function. TFColor is already imported. self.tfunc is the current transfer function.
    def render_compositing_mouse(self, view_matrix: np.ndarray, volumes, image_size: int, image: np.ndarray, colors):

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # w vector. See documentation in parent's class
        w_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2
        print(np.sum(image))
        # Center of the volume (3-dimensional)
        volume_center = [volumes[0].dim_x / 2, volumes[0].dim_y / 2, volumes[0].dim_z / 2]
        volume_maxima = [volume.get_maximum() for volume in volumes]

        tfuncs = []
        for index, maxima in enumerate(volume_maxima):
            tf = TransferFunction()
            tf.init(0, maxima)
            tf.add_control_point(0, .0, .0, .0, .0)
            tf.add_control_point(maxima, colors[index][0], colors[index][1], colors[index][2], 1.0)
            tfuncs.append(tf)

        # Define a step size to make the loop faster
        step = 1 if self.interactive_mode else 1
        # img_neg = -1 * image_center

        # self.tfunc.set_test_function()
        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                colorAll = TFColor(0, 0, 0, 1)
                for k in range(int(-image_size / 2), int(image_size / 2), 5):

                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         w_vector[0] * (k) + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         w_vector[1] * (k) + volume_center[1]

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         w_vector[2] * (k) + volume_center[2]
                    if (voxel_coordinate_x < 0 or voxel_coordinate_x >= volumes[0].dim_x or \
                            voxel_coordinate_y < 0 or voxel_coordinate_y >= volumes[0].dim_y or \
                            voxel_coordinate_z < 0 or voxel_coordinate_z >= volumes[0].dim_z):
                        continue

                    denom = 0.
                    r = g = b = a = 0.
                    for index, volume in enumerate(volumes):
                        value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                        voxel_coordinate_z)

                        if (value == 0):
                            continue
                        # if (value > max):
                        #   max = value

                        denom += 1.
                        vol_color = tfuncs[index].get_color(value)
                        r += vol_color.r
                        g += vol_color.g
                        b += vol_color.b
                        a += vol_color.a
                    if denom == 0:
                        continue
                    color = TFColor(r / denom, g / denom, b / denom, a / denom)
                    # Normalize value to be between 0 and 1
                    colorAll.r = color.a * color.r + (1 - color.a) * colorAll.r
                    colorAll.g = color.a * color.g + (1 - color.a) * colorAll.g
                    colorAll.b = color.a * color.b + (1 - color.a) * colorAll.b
                    # colorAll.a = color.a

                # Compute the color value (0...255)
                red = math.floor(colorAll.r * 255) if colorAll.r < 1.0 else 255
                green = math.floor(colorAll.g * 255) if colorAll.g < 1.0 else 255
                blue = math.floor(colorAll.b * 255) if colorAll.b < 1.0 else 255
                alpha = math.floor(colorAll.a * 255) if colorAll.a < 1.0 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha
                # print("i j k:", i)
        # print("value", value, "max", max)
        print("DONE")

    def render_mip_1(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray, default=0.):
        # Clear the image

        #self.clear_image()
        max_list = set()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # w vector. See documentation in parent's class
        w_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2
        print(np.sum(image))
        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 1 if self.interactive_mode else 1
        img_neg = -1 * image_center
        z_step = 5

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                max = 0
                for k in range(int(img_neg), int(image_center), z_step):

                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         w_vector[0] * (k) + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         w_vector[1] * (k) + volume_center[1]

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         w_vector[2] * (k) + volume_center[2]
                    # Get voxel value
                    value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                    voxel_coordinate_z)
                    if (value > max):
                        max = value
                if max == 0:
                    continue
                # Normalize value to be between 0 and 1
                red = default
                green = red
                blue = red
                alpha = 0.7 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 1.0 else 255
                green = math.floor(green * 255) if green < 1.0 else 255
                blue = math.floor(blue * 255) if blue < 1.0 else 255
                alpha = math.floor(alpha * 255) if alpha < 1.0 else 255
                #if i % 10 == 1 and j % 10 == 1 and red < 0: print("inside:", i, j, red)

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha
                #if max != 0: max_list.add((max, red, image[(j * image_size + i) * 4]))

                # print("==============", i)
        print(max_list)

    def render_mip_energy(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray, color):

        max_list = set()
        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # w vector. See documentation in parent's class
        w_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2
        print(np.sum(image))

        # volume = volumes[0]
        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]

        # TODO
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 1 if self.interactive_mode else 1
        img_neg = -1 * image_center
        z_step = 5

        for i in range(int(img_neg), image_size, step):
            for j in range(0, image_size, step):
                max = 0
                for k in range(0, int(image_center), z_step):

                    # for volume in volumes:
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         w_vector[0] * (k) + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         w_vector[1] * (k) + volume_center[1]

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         w_vector[2] * (k) + volume_center[2]
                    # Get voxel value
                    value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                    voxel_coordinate_z)
                    if (value > max):
                        max = value
                if max == 0 or max == -1:
                    continue

                # Normalize value to be between 0 and 1
                red = color[0] * (max / volume_maximum)
                green = color[1] * (max / volume_maximum)
                blue = color[2] * (max / volume_maximum)
                alpha = 0.3 if max > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 1.0 else 255
                green = math.floor(green * 255) if green < 1.0 else 255
                blue = math.floor(blue * 255) if blue < 1.0 else 255
                alpha = math.floor(alpha * 255) if alpha < 1.0 else 255
                if i % 10 == 1 and j % 10 == 1 and red < 0: print("inside:", i, j, red)

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha
                if max != 0: max_list.add((max, red, image[(j * image_size + i) * 4]))
        print(max_list)

    def render_mip_energies(self, view_matrix: np.ndarray, volumes, image_size: int, image: np.ndarray, colors):

        max_list = set()
        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # w vector. See documentation in parent's class
        w_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2
        print(np.sum(image))

        volume = volumes[0]
        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]

        # TODO
        volume_maxima = [volume.get_maximum() for volume in volumes]

        # Define a step size to make the loop faster
        step = 1 if self.interactive_mode else 1
        img_neg = -1 * image_center
        z_step = 5

        for i in range(int(img_neg), image_size, step):
            for j in range(0, image_size, step):
                max = [0] * len(volumes)
                for k in range(0, int(image_center), z_step):

                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         w_vector[0] * (k) + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         w_vector[1] * (k) + volume_center[1]

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         w_vector[2] * (k) + volume_center[2]

                    # Get voxel value
                    values = [trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                      voxel_coordinate_z) for volume in volumes]
                    for c in range(len(volumes)):
                        if (values[c] > max[c]):
                            max[c] = values[c]

                if sum(max) == 0 or max == -1:
                    continue

                red = green = blue = alpha = 0.
                denom = 0.
                for c in range(len(volumes)):
                    if max[c] == 0 or max[c] == -1: continue
                    # Normalize value to be between 0 and 1
                    red += colors[c][0] * (max[c] / volume_maxima[c])
                    green += colors[c][1] * (max[c] / volume_maxima[c])
                    blue += colors[c][2] * (max[c] / volume_maxima[c])
                    denom += 1.
                r = image[(j * image_size + i) * 4] if image[(j * image_size + i) * 4] >= 0 else image[(
                                                                                                                   j * image_size + i) * 4] + 255
                b = image[(j * image_size + i) * 4 + 1] if image[(j * image_size + i) * 4 + 1] >= 0 else image[(
                                                                                                                           j * image_size + i) * 4 + 1] + 255
                g = image[(j * image_size + i) * 4 + 2] if image[(j * image_size + i) * 4 + 2] >= 0 else image[(
                                                                                                                           j * image_size + i) * 4 + 2] + 255
                red += 0.1 * r / 255
                blue += 0.1 * g / 255
                green += 0.1 * b / 255
                # denom += 1

                red /= denom
                blue /= denom
                green /= denom
                alpha = 1.0
                # if alpha<0.1: continue
                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 1.0 else 255
                green = math.floor(green * 255) if green < 1.0 else 255
                blue = math.floor(blue * 255) if blue < 1.0 else 255
                alpha = math.floor(alpha * 255) if alpha < 1.0 else 255
                if i % 10 == 1 and j % 10 == 1 and red < 0: print("inside:", i, j, red)

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha
                # max_list.add((max, red, image[(j * image_size + i) * 4]))
                # if i==30 and red+blue+green+alpha>0:
                # print(red, green, blue, alpha)
                # print(image[(j * image_size + i) * 4] , image[(j * image_size + i) * 4+1] , image[(j * image_size + i) * 4+2] , image[(j * image_size + i) * 4+3] )
        print(max_list)

    def render_mouse_brain(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict,
                           image_size: int, image: np.ndarray):
        # TODO: Implement your code considering these volumes (annotation_volume, and energy_volumes)
        self.render_mip_mouse(view_matrix, annotation_volume, image_size, image, default=0.8)
        #self.annotation_gradient_volume
        colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.], [0., 1., 1.], [1., 0.5, 0.], [1., 0., 0.5],
                  [1., 0., 1.], [1.0, 1.0, 0.5], [0.5, 0.5, 0.5]]
        #if not energy_volumes == {}:
            #self.render_compositing_mouse(view_matrix, list(energy_volumes.values()), image_size, image, colors)


class GradientVolumeImpl(GradientVolume):
    # TODO: Implement gradient compute function. See parent class to check available attributes.
    def compute(self):
        pass


import numpy as np
import matplotlib.pyplot as plt
# TODO BUGFIX THIS
def display_data(data, example_count):
    data = np.array(data)

    m, n = data.shape
    m = example_count
    example_width = np.round(np.sqrt(data.shape[1]))
    example_height = (n / example_width)
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)

    # Debug
    """
    print("Height - %i" % example_height)
    print("Width - %i" % example_width)
    print("Shape of data - %s" % str(data.shape))
    """
    # Padding detween images
    pad = 1
    # Blank array
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                       pad + display_cols * (example_width + pad)))
    curr_ex = 0
    for j in range(int(display_rows)):
        for i in range(int(display_cols)):
            if curr_ex >= m:
                break

            max_val = max(np.abs(data[curr_ex, :]))
            for k in range(1, int(example_height)):
                for l in range(1, int(example_width)):
                    cur_height = pad + j * (example_height + pad) + k
                    cur_width = pad + i * (example_width + pad) + l
                    cur_ex_pixel = (l-1) * example_height + k - 1
                    display_array[cur_height, cur_width] = data[curr_ex, cur_ex_pixel] / max_val

            curr_ex += 1

        if curr_ex >= m:
            break
    imgplot = plt.imshow(display_array)
    #imgplot.set_cmap('Greys')
    plt.show()

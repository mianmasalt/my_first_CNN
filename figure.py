import matplotlib.pyplot as plt

def CNN_plot(X,Y,table_name,x_label,y_label,save_path=None):
    plt.figure()
    for y in Y:
        plt.plot(X,y)
    plt.title(table_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig(f"{table_name}.png")
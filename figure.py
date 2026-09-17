import matplotlib.pyplot as plt

def CNN_plot(X,Y,curve_names,table_name,x_label,y_label,path=None):
    plt.figure()
    for i, y in enumerate(Y):
        plt.plot(X, y, label=curve_names[i])
    plt.title(table_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    save_path='my_CNN/my_first_CNN/result_fig/'+path
    if path:
        plt.savefig(save_path)
    else:
        plt.savefig(f"{table_name}.png")

    plt.show()
    plt.close()
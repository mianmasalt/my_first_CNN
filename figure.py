import matplotlib.pyplot as plt
from pathlib import Path

def CNN_plot(X,Y,curve_names,table_name,x_label,y_label,filename):
    plt.figure()
    for i, y in enumerate(Y):
        plt.plot(X, y, label=curve_names[i])
    plt.title(table_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    save_dir=Path(__file__).resolve().parent/"result_fig"
    save_dir.mkdir(parents=True,exist_ok=True)

    save_path=save_dir/filename
    plt.savefig(save_path)

    plt.show()
    plt.close()
import matplotlib.pyplot as plt

def plot_matrix(A, xlabel='', ylabel='', title='', cmap='spring', extent=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(A.T, cmap=cmap, aspect='auto', origin='lower', extent=extent)
    plt.colorbar(label='Value')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
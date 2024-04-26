import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("sensor_data.csv")

sensors = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']

x_range = df['X'].min(), df['X'].max()
y_range = df['Y'].min(), df['Y'].max()
sensor_range = df[sensors].min().min(), df[sensors].max().max()

for sensor in sensors:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['X'], df['Y'], df[sensor])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Resistance')
    ax.set_title(f'{sensor} 3D Plot')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(sensor_range)

    plt.show()

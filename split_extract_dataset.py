import pandas as pd
import csv

def extractXYandForce():
    data = pd.read_csv('sensor_data.csv', header=None)

    data = data.iloc[1:]
    xyCoordinates = data.iloc[:, [8, 9]].values.tolist()
    forces = data.iloc[:, 10].tolist()

    uniqueXyCoordinates = []
    uniqueForces = []

    seenXyCoordinates = set()
    seenForces = set()

    for coord, force in zip(xyCoordinates, forces):
        coordTuple = tuple(coord)
        if coordTuple not in seenXyCoordinates:
            seenXyCoordinates.add(coordTuple)
            coord[0] = float(coord[0])
            coord[1] = float(coord[1])
            uniqueXyCoordinates.append(coord)
        if force not in seenForces:
            seenForces.add(force)
            uniqueForces.append(float(force))

    print(uniqueForces)
    print(uniqueXyCoordinates)
    print()
    return uniqueXyCoordinates, uniqueForces
def calculateAverages():
    data = pd.read_csv('sensor_data.csv')
    sensorReadings = data.iloc[:, :8].values.tolist()
    sensorReadings = list(map(list, zip(*sensorReadings)))
    num_rows = len(sensorReadings[1])
    num_columns = len(sensorReadings)


    noDuplicateReadings = [[], [], [], [], [], [], [], []]

    for i in range(0, num_columns):
        for j in range(0, num_rows, 20):
            resAverage = sum(sensorReadings[i][j:j + 20]) / len(sensorReadings[i][j:j + 20])
            noDuplicateReadings[i].append(resAverage)

    transposeNewReadings = list(map(list, zip(*noDuplicateReadings)))

    truncatedData = []

    for row in transposeNewReadings:
        truncatedRow = [round(value, 9) for value in row]
        truncatedData.append(truncatedRow)
    print(len(truncatedData))
    print(len(truncatedData[0]))
    print(truncatedData)
    return truncatedData

def writeToCSV():
    data = calculateAverages()
    csvFilePath = 'avg_sensor_data.csv'

    with open(csvFilePath, 'w', newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(data)

    print('Successful')

coords = extractXYandForce()[0]

coordinates_list = extractXYandForce()[0]

numbered_coordinates = [(i, coord[0], coord[1]) for i, coord in enumerate(coordinates_list)]

sorted_coordinates = sorted(numbered_coordinates, key=lambda x: x[1])

csv_filename = 'output_coordinates.csv'
with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Number', 'X Coordinate', 'Y Coordinate'])
    for row in sorted_coordinates:
        csv_writer.writerow(row)

print(f'Coordinates have been written to {csv_filename}')

extractXYandForce()

calculateAverages()


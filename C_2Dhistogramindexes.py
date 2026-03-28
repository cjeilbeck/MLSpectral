from A_functions import read_data, indicesmav, problemtype
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
plt.rcParams.update({'font.size': 14})
import matplotlib.patches as patches

"""  ids['Green'] = M(560) 
    ids['Red'] = M(650)
    ids['Red-edge'] = M(730)
    ids['NIR'] = M(860, width=26)
    ids['NDVI']= (M(860, width=26)-M(650))/(M(860, width=26)+M(650))
    ids['GNDVI']=(M(860)-M(560))/(M(860)+M(560))
    ids['OSAVI']= (M(860, width=26)-M(650))/(M(860, width=26)+M(650)+0.16)
    ids['LCI']= (M(860, width=26)-M(730))/(M(860, width=26)+M(650))
    ids['NDRE']= (M(860, width=26)-M(730))/(M(860, width=26)+M(730))"""


BINARY = True
GUM = False
        


#parameters now wrong after fixed GNDVI
def binaryclassify(row):

    x = row['NDRE']
    y = row['GNDVI']

    if x <= 0.428 and y <= 0.585:
        return 'Gum'
    elif 0.444 <= x <= 0.453 and 0.565 >= y >= 0.525:
        return 'Gum'
    elif 0.452 <= x <= 0.47 and 0.615 >= y >= 0.5822:
        return 'Gum'
    elif 0.4632 <= x <= 0.4718 and 0.6795 >= y >= 0.65588:
        return 'Gum'
    else:
        return 'Other'
    
filename = 'CSVfiles/datacalibrated.csv'

x,y = read_data(filename)
indices=indicesmav(x)
plot = indices.copy()
plot['leaf_type']=y

index1 = 'NDRE'
index2 = 'GNDVI'

plot['leaf_type']=y

if BINARY:

    plot = problemtype(plot, 'binary')
    print("classifiying")
    plot['predicted'] = plot.apply(binaryclassify, axis=1)
    accuracy = accuracy_score(plot['leaf_type'], plot['predicted'])
    print(f"Classification Accuracy: {accuracy:.4f}")
    classification_report = classification_report(plot['leaf_type'], plot['predicted'])
    print("Classification Report:")
    print(classification_report)

if GUM:
    plot = problemtype(plot, 'gum')

fig = plt.figure(figsize=(10, 7))


xx = index1
yy = index2

sns.scatterplot(data=plot, x=xx, y=yy, hue='leaf_type', palette='Set2', edgecolor='black', alpha=0.7)
plt.xlabel(xx)
plt.ylabel(yy)

box1 = [0, 0, 0.428, 0.585] 
box2 = [0.444, 0.525, 0.453-0.444, 0.565-0.525]
box3 = [0.452, 0.5822, 0.47-0.452, 0.615-0.5822]
box4 = [0.4632, 0.65588, 0.4718-0.4632, 0.6795-0.65588]

boxes = [box1, box2, box3, box4]
ax = plt.gca()

for b in boxes:
    rect = patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Gum Area' if b == boxes[0] else "")
    ax.add_patch(rect)

plt.legend()

plt.show()
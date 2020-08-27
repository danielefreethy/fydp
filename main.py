import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set(style='white')
import pandas as pd
import glob
import sys
import numpy as np

# Check input
if "-pca" in sys.argv and "-heatmap" in sys.argv:
    print("Choose only pca or heatmap")
    sys.exit()

if "-pca" in sys.argv and "-batch_num" not in sys.argv and "-fault_ref" not in sys.argv:
    print("Choose '-batch-num' or '-fault_ref' as the target for PCA")
    sys.exit()

if "-pca" not in sys.argv and "-heatmap" not in sys.argv and "-test" not in sys.argv:
    print("Choose a valid option")
    sys.exit()

# Load all csvs
all_files = glob.glob("*.csv")
li = []
filenames = []

hun = []
step = 1150
for i in range(100):
    temp = [i+1] * step
    hun += temp

counter = 0
# Load all the datasets into one
print("Files Read: ")
for filename in all_files:
    print(filename)
    filenames.append(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    df = df.drop(
        columns=['RPM', 'X_offline', 'Viscosity_offline', 'P_offline', 'NH3_offline', 'PAA_offline', 'Time (h)', 'Control_ref', 'NH3_shots', 'Batch_ref', 'PAT_ref'])
    if len(df.columns) == 27:
        df = df.drop(
            columns=['Batch ID', 'Batch #']
        )

    columns = df.columns.values
    new_labels = []
    counter3 = 0
    for i in range(1150):
        counter3 += 0.2
        temp = str(round(counter3, 1)) + ' - ' + columns
        new_labels.extend(temp)

    counter2 = 0
    partial_df = []
    while counter2 < len(df.index):
        tdf = df[counter2:counter2 + 1150].values.flatten()
        partial_df.append(tdf)
        counter2 += 1150

    #temp = [counter] * len(df.index)
    df = pd.DataFrame(partial_df)
    df.columns = new_labels
    # Tells us which file/batch the data is from
    #df['File_Num'] = temp
    #batch_nums = [x + 100 * counter for x in hun]
    #df['Batch_Num'] = batch_nums
    #counter += 1
    li.append(df)
print()

data = pd.concat(li, axis=0, ignore_index=True)
print("Data shape: " + str(data.shape))
# Features being dropped
features = data.columns.tolist()
# Separate dataset from target
x = data.loc[:, features].values
#y = data.loc[:, ['Fault_ref']].values
#y2 = data.loc[:, ['Batch_Num']].values
# Standardizing the data
x = StandardScaler().fit_transform(x)

if "-heatmap" in sys.argv:
    ax = sns.heatmap(data.corr())
    ax.grid()
    plt.show()

if "-pca2" in sys.argv:
    variances = []
    n_comp = np.arange(30)
    for i in range(30):
        pca = PCA(n_components=i)
        principalComponents = pca.fit_transform(x)
        cevr = sum(pca.explained_variance_ratio_)
        print(str(i) + " component(s) - " + "PCA Variance: ", cevr)
        variances.append(cevr)
    plt.plot(n_comp, variances)
    plt.ylabel('Cumulative explained variance ratio')
    plt.xlabel('n_components')
    plt.title('CEVR by Number of Components')
    plt.grid(True)
    plt.show()

    # Plotting 2 components
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2'])
    print(principalDf)
    x = principalDf['pc1'].values
    y = principalDf['pc2'].values
    plt.scatter(x, y)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


if "-pca" in sys.argv:
    # PCA Stuff
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    print("PCA Variance: ", pca.explained_variance_ratio_)
    principalDf = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2'])

    finalDf = pd.concat([principalDf, data[['Fault_ref']]], axis=1)
    finalDf2 = pd.concat([principalDf, data[['Batch_Num']]], axis=1)

    # Pairplot the PCA result, not sure if this is needed
    #g = sns.pairplot(finalDf, hue="Fault_ref")


    # Plot results
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [1, 0]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#FFE4E1', '#eeefff']

    if "-batch_num" in sys.argv:
        for i in range(len(li)):
            temp_df = finalDf2[finalDf2['Batch_Num'] == i]
            ax.scatter(temp_df['pc1'], temp_df['pc2'], c=colors[i], s=5)
            ax.legend(filenames)

    if "-fault_ref" in sys.argv:
        fault_df = finalDf[finalDf['Fault_ref'] == 1]
        good_df = finalDf[finalDf['Fault_ref'] == 0]
        print("Fault shape: ", fault_df.shape)
        print("Good shape: ", good_df.shape)
        ax.scatter(fault_df['pc1']
                   , fault_df['pc2']
                   , c='r'
                   , s=50)
        ax.scatter(good_df['pc1']
                   , good_df['pc2']
                   , c='g'
                   , s=50)

        ax.legend(targets)
    ax.grid()
    plt.show()

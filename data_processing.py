def load_data(data_csv):
        """
        Load the data from a csv file. Just return the X and y, splitting into
        train and test data will be handled elsewhere.
        """
        # Load the data
        data = pd.read_csv(data_csv, index_col=0)

        data['outcome'] = data.apply(lambda x: 0 if x['winner'] == 'Away' else 1, axis=1)

        y = data['outcome']

        X = data.drop(['winner', 'outcome','winning_abbr'], axis=1)
        # Take care of any NaN values
        X = X.fillna(0)
        return X.to_numpy(), y.to_numpy()

def scale_data(x_raw):
    scaler = StandardScaler().fit(x)
    X = scaler.transform(x)
    return X



x,y= load_data(data_path)
scaler = StandardScaler().fit(x)
#print("b4")
X = scaler.transform(x)
#print("after")
#print(X)
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents)
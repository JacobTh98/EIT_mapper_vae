class DataGenerator_ref(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, path, batch_size=32, dim=1024, shuffle=True):
        'Initialization'
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X
    
    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, self.dim, 1))
        y = np.empty((self.batch_size, self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            tmp = np.load('{0:s}/sample_{1:06d}.npz'.format(self.path, ID), allow_pickle=True)
            mesh_obj = tmp['mesh_obj'].tolist()
            X[i,] = np.expand_dims(mesh_obj['perm'], axis=1)
            y[i,] = np.expand_dims(mesh_obj['perm'], axis=1)
            
            X[i,] = X[i,] / 25 
            y[i,] = y[i,] / 25 

        return X, y

class DataGenerator_elsig_ref(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, path, batch_size=32, dim=1024, shuffle=True):
        'Initialization'
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, self.dim, 1))
        y = np.empty((self.batch_size, 192, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            tmp = np.load('{0:s}/sample_{1:06d}.npz'.format(self.path, ID), allow_pickle=True)
            mesh_obj = tmp['mesh_obj'].tolist()
            X[i,] = np.expand_dims(mesh_obj['perm'], axis=1)
            y[i,] = np.expand_dims(tmp['electrode_signals'], axis=1)
            
            # normalization
            X[i,] = X[i,] / 25
            y[i,] = y[i,] / 25

        return X, y
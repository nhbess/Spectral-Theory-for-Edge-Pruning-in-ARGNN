import gzip
import os
import pickle

def save_model(m, filename, foldername):
    if not os.path.exists(f'{foldername}'):
        os.makedirs(f'{foldername}')
    
    with open (f'{foldername}/{filename}.pkl', 'wb') as file:
        pickle.dump(m, file)
    
    if os.path.exists(f'{foldername}/{filename}.pkl'):
        with open (f'{foldername}/{filename}.pkl', 'rb') as file:
            with gzip.open(f'{foldername}/{filename}.pkl.gz', 'wb') as filez:
                filez.write(file.read())
        os.remove(f'{foldername}/{filename}.pkl')
        print('Model saved and compressed')
    else:
        print('Something went wrong here!')

    size = os.path.getsize(f'{foldername}/{filename}.pkl.gz')
    if size > 100000000:
        print(f'WARNING: Model size is {size/1000000}MB, Git will complain!')


def load_model(filename, foldername):
    if os.path.exists(f'{foldername}/{filename}.pkl'):
        with open (f'{foldername}/{filename}.pkl', 'rb') as file:
            return pickle.load(file)
    
    elif os.path.exists(f'{foldername}/{filename}.pkl.gz'):
        import gzip
        with gzip.open(f'{foldername}/{filename}.pkl.gz', 'rb') as filez:
            return pickle.load(filez)
    
    else:
        raise Exception('Model not found')
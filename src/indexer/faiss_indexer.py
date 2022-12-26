import faiss



def get_faiss_indexer():
    '''


    Return
    ------
    '''
    indexer = faiss.IndexFlatL2(256)

    return indexer
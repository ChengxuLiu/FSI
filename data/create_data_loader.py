from .dataset_data_loader import DatasetDataLoader


def CreateDataLoader(opt):
    data_loader = DatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataLoader_train(opt):
    opt.phase='train'
    data_loader_train = DatasetDataLoader()
    print(data_loader_train.name())
    data_loader_train.initialize(opt)

    opt.phase='test'
    opt.batchSize=1
    opt.nThreads=1
    data_loader_test= DatasetDataLoader()
    print(data_loader_test.name())
    data_loader_test.initialize(opt)

    opt.phase='train'
    return data_loader_train,data_loader_test

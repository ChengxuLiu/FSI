import time
from options.train_options import TrainOptions
from data.create_data_loader import CreateDataLoader
from model.models import create_model
from util.visualizer import Visualizer



def train(opt, data_loader_train, model, visualizer):
    dataset_train = data_loader_train.load_data()
    dataset_train_size = len(data_loader_train)
    print('#training images = %d' % dataset_train_size)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset_train):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                results = model.get_current_visuals()
                visualizer.display_current_results(results, epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter, time.time() - epoch_start_time))
        model.warmup_scheduler()


opt = TrainOptions().parse()
data_loader_train = CreateDataLoader(opt)
model = create_model(opt)
visualizer = Visualizer(opt)
train(opt, data_loader_train, model, visualizer)



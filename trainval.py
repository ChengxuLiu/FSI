import time
from options.train_options import TrainOptions
from data.create_data_loader import CreateDataLoader_train
from model.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
import lpips
from DISTS_pytorch import DISTS
import time


def trainval(opt, data_loader_train,data_loader_test, model, visualizer):
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


        print('begin test' )
        perceptual_eval = True
        if perceptual_eval:
            loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
            # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
            dists = DISTS()
        eLPIPS = 0.0
        eDISTS = 0.0
        ePSNR = 0.0
        eSSIM = 0.0
        dataset_test = data_loader_test.load_data()
        dataset_test_size = len(data_loader_test)
        print('#test images = %d' % dataset_test_size)
        for i, data in enumerate(dataset_test):
            model.set_input(data)
            start = time.time()
            model.test()
            end = time.time()
            # print('inference time', end-start)
            eval_data = model.get_eval_data()
            visuals = model.get_current_visuals()

            x = eval_data['x']
            y = eval_data['y']
            output = eval_data['x_hat']

            img_path = model.get_image_paths()
            # print('process image... %s' % img_path)

            if perceptual_eval:
                tensors = model.get_tensor_raw_data()
                # for DISTS [0, 1]
                tensor_data = model.get_tensor_raw_data()
                x_tensor = tensor_data['x']
                y_tensor = tensor_data['y']
                output_tensor = tensor_data['x_hat']
                eDISTS += dists(x_tensor, output_tensor)

                # for LIPIPS [-1, 1]
                x_tensor = x_tensor * 2.0 - 1
                y_tensor = y_tensor * 2.0 - 1
                output_tensor = output_tensor * 2.0 - 1
                eLPIPS += loss_fn_alex(x_tensor, output_tensor)

            ePSNR += PSNR(x, output)
            eSSIM += SSIM(x, output)
            # visualizer.display_current_results(visuals, 1)
            # visualizer.save_images(web_dir, visuals, img_path)
        avgPSNR = ePSNR/dataset_test_size
        avgSSIM = eSSIM/dataset_test_size
        print('avgPSNR : %f' % avgPSNR)
        print('avgSSIM : %f' % avgSSIM)
        if perceptual_eval:
            avgLPIPS = eLPIPS/dataset_test_size
            avgDISTS = eDISTS/dataset_test_size
            print('avgLPIPS : %f' % avgLPIPS)
            print('avgDISTS : %f' % avgDISTS)
        del loss_fn_alex
        del dists
  

  
opt = TrainOptions().parse()
data_loader_train,data_loader_test = CreateDataLoader_train(opt)
model = create_model(opt)
visualizer = Visualizer(opt)
trainval(opt, data_loader_train,data_loader_test, model, visualizer)



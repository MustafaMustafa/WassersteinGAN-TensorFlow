import os
import time
import tensorflow as tf
from utils import * 

def train_wasserstein(sess, gan, data, config):
    """Train wasserstein GAN"""

    d_optim = tf.train.RMSPropOptimizer(config.lrD).minimize(gan.d_loss, var_list=gan.d_vars, global_step=gan.global_step)
    g_optim = tf.train.RMSPropOptimizer(config.lrG).minimize(gan.g_loss, var_list=gan.g_vars)
    cap_d_vars_ops = [val.assign(tf.clip_by_value(val, -config.clamp, config.clamp)) for val in gan.d_vars]

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    g_sum = tf.summary.merge([gan.z_sum, gan.d__sum, gan.G_sum, gan.d_loss_fake_sum, gan.g_loss_sum])
    d_sum = tf.summary.merge([gan.z_sum, gan.d_sum, gan.d_loss_real_sum, gan.d_loss_sum])
    writer = tf.summary.FileWriter("./logs/"+config.tensorboard_run, sess.graph)

    sample_z = gan.z_gen(shape=(gan.sample_size , gan.z_dim))
    sample_images = get_batch_images(0, data, config)
        
    start_time = time.time()

    if load(sess, gan, config):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    for epoch in range(config.epoch):

        idx = 0
        gen_iterations = 0
        batch_idxs = min(len(data), config.train_size) // config.batch_size
        while idx < batch_idxs:
            
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                D_iters = 100
            else:
                D_iters = config.nc

            # Update D network
            for i in range(0, D_iters):
                sess.run(cap_d_vars_ops)

                batch_images = get_batch_images(idx, data, config)
                batch_z = gan.z_gen(shape=(gan.batch_size , gan.z_dim)).astype(np.float32)
                _, summary_str = sess.run([d_optim, d_sum],feed_dict={ gan.images: batch_images, gan.z: batch_z })
                writer.add_summary(summary_str, gan.global_step.eval())
                idx += 1

            # Update G network
            batch_z = gan.z_gen(shape=(gan.batch_size , gan.z_dim)).astype(np.float32)
            _, summary_str = sess.run([g_optim, g_sum], feed_dict={ gan.z: batch_z })
            writer.add_summary(summary_str, gan.global_step.eval())
            gen_iterations += 1

            errD_fake = gan.d_loss_fake.eval({gan.z: batch_z})
            errD_real = gan.d_loss_real.eval({gan.images: batch_images})
            errG = gan.g_loss.eval({gan.z: batch_z})

            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))

            if np.mod(gen_iterations, 100) == 1:
                samples, d_loss, g_loss = sess.run([gan.sampler, gan.d_loss, gan.g_loss], 
                                                        feed_dict={gan.z: sample_z, gan.images: sample_images})

                save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

        # save a checkpoint every epoch
        save(sess, gan, config)

def save(sess, gan, config):
    model_name = gan.model_name + "_Wasserstein.model-epoch"
    model_dir = "%s_%s_%s" % (config.dataset, config.batch_size, config.output_size)
    checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    gan.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=gan.global_step)

def load(sess, gan, config):
    print(" [*] Reading checkpoints...")

    model_dir = "%s_%s_%s" % (config.dataset, config.batch_size, config.output_size)
    checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        gan.saver.restore(sess, os.path.join(config.checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False

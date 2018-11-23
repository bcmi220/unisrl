#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())
import json
import time
import random
import shutil

import numpy as np
import tensorflow as tf

from lsgn_data import LSGNData
from lsgn_evaluator import LSGNEvaluator
from srl_model import SRLModel
import util

def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)

if __name__ == "__main__":
  if len(sys.argv) > 1:
    name = sys.argv[1]
  else:
    name = os.environ["EXP"]
  config = util.get_config("experiments.conf")[name]
  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]

  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
  util.print_config(config)

  # if "GPU" in os.environ:
  #   gpus = [int(g) for g in os.environ["GPU"].split(",")]
  #   util.set_gpus(*gpus)
  # else:
  #   util.set_gpus()
  if len(sys.argv) > 3 and sys.argv[2] == '-gpu':
	  util.set_gpus(sys.argv[3])

  data = LSGNData(config)
  model = SRLModel(data, config)
  evaluator = LSGNEvaluator(config, quiet=True)
  saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()

  log_dir = config["log_dir"]
  assert not ("final" in name)  # Make sure we don't override a finalized checkpoint.
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  # Create a "supervisor", which oversees the training process.
  sv = tf.train.Supervisor(logdir=log_dir,
                           init_op=init_op,
                           saver=saver,
                           global_step=model.global_step,
                           save_model_secs=120)

  max_f1 = 0

  # The supervisor takes care of session initialization, restoring from
  # a checkpoint, and closing when done or an error occurs.
  with sv.managed_session() as session:
    data.start_enqueue_thread(session)
    accumulated_loss = 0.0
    initial_time = time.time()
    while not sv.should_stop():
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print "[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second)
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step % eval_frequency == 0:
        eval_summary, f1, task_to_f1 = evaluator.evaluate(
            session, data, model.predictions, model.loss)

        if f1 > max_f1:
          max_f1 = f1
          ckpt = tf.train.get_checkpoint_state(log_dir)
          tmp_checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
          copy_checkpoint(ckpt.model_checkpoint_path, tmp_checkpoint_path)
        
        print "Current: {:.2f} max combined F1: {:.2f}".format(f1, max_f1)

  # Ask for all the services to stop.
  sv.stop()

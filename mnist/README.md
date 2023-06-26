# python-playground/mnist

```sh
$ python nn.py
Epoch: 0, Accuracy: 0.7754
Epoch: 1, Accuracy: 0.9152
Epoch: 2, Accuracy: 0.9299
Epoch: 3, Accuracy: 0.9384
Epoch: 4, Accuracy: 0.9433
Epoch: 5, Accuracy: 0.9470
Epoch: 6, Accuracy: 0.9507
Epoch: 7, Accuracy: 0.9543
Epoch: 8, Accuracy: 0.9562
Epoch: 9, Accuracy: 0.9574
Epoch: 10, Accuracy: 0.9601
Epoch: 11, Accuracy: 0.9611
Epoch: 12, Accuracy: 0.9631
Epoch: 13, Accuracy: 0.9643
Epoch: 14, Accuracy: 0.9657
Epoch: 15, Accuracy: 0.9669
Epoch: 16, Accuracy: 0.9680
Epoch: 17, Accuracy: 0.9686
Epoch: 18, Accuracy: 0.9691
Epoch: 19, Accuracy: 0.9702
Test accuracy: 0.9568

# took 2m14s
```

```sh
$ python nn_lightning.py
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 2 processes
----------------------------------------------------------------------------------------------------

You are using a CUDA device ('NVIDIA RTX A6000') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]

  | Name      | Type               | Params
-------------------------------------------------
0 | train_acc | MulticlassAccuracy | 0
1 | valid_acc | MulticlassAccuracy | 0
2 | test_acc  | MulticlassAccuracy | 0
3 | model     | Sequential         | 25.8 K
-------------------------------------------------
25.8 K    Trainable params
0         Non-trainable params
25.8 K    Total params
0.103     Total estimated model params size (MB)
Epoch 9: 100%|███████| 860/860 [00:05<00:00, 162.25it/s, v_num=1, train_loss_step=1.460, valid_loss_step=1.460, valid_loss_epoch=1.520, valid_acc=0.946, train_loss_epoch=1.510]`Trainer.fit` stopped: `max_epochs=10` reached.
Epoch 9: 100%|███████| 860/860 [00:05<00:00, 162.07it/s, v_num=1, train_loss_step=1.460, valid_loss_step=1.460, valid_loss_epoch=1.520, valid_acc=0.946, train_loss_epoch=1.510]

# took 1m1s
```

### train: 
python train.py --gcn_bool --addaptadj --randomadj --data data/bb-task-1-5 --adjdata data/sensor_graph/adj_BB.pkl --num_nodes 6 --in_dim 2
python train.py --gcn_bool --addaptadj --randomadj --data data/tg-task-1-0 --adjdata data/sensor_graph/adj_mx_TG_new.pkl --num_nodes 19 --in_dim 2 --epochs 50

### wandb log to upload:
1. (not dichotomy)wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240406_134108-jfqaee46
2. wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240408_111513-uo0ux5dh
3. wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240408_121230-wh114t03
4. wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240408_122445-wg6lbbae

### (extra random task seq) wandb log to upload:
5.  (exp2) wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240422_233920-lq5u15hl
6.  (exp3) wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240422_235835-5c8lxjbb
7.  (二分0-1) wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240423_104051-q3nmytm0
8.  (二分10-100) wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240423_110155-yrtcljv2
9.  (interval=5) wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240423_123956-vcmsjexq
10. (interval=10) wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240423_124037-41etjzir
11. (interval=20) wandb sync /home/imrs/CaoBo/Graph-WaveNet/wandb/offline-run-20240423_125043-pom3f5ba

### test: 
python test.py --gcn_bool --addaptadj --randomadj --data data/tg-task-1-5 --adjdata data/sensor_graph/adj_mx_TG_new.pkl --num_nodes 19 --in_dim 2 --checkpoint pth_files/tg_5_best_5.72.pth

from selfplaylab.game.go import CaptureGoState
import torch, time
import numpy as np

torch.set_num_threads(1)

game_class = CaptureGoState

for run in [0, 1]:
    if run == 1:
        torch.set_num_threads(1)

    for cuda in [False, True]:
        net = game_class.create_net(cuda=cuda)
        print(net.device)
        st = game_class()
        encoded_s = torch.from_numpy(st.encoded()).unsqueeze(0).float().to(net.device)
        inp = encoded_s
        net(inp)
        with torch.no_grad():
            for i in range(12):
                start = time.time()
                net(inp)
                t = time.time() - start
                print(inp.shape[0], ":", t, "=", 1000 * t / inp.shape[0], "ms/sample")
                inp = torch.cat([inp, inp], dim=0)
                inp = inp + torch.randn(*inp.shape).float().to(net.device)

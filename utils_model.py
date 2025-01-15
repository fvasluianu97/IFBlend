import torch
from ifblend import IFBlend


def get_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "ifblend":
        return IFBlend(16, device=device, use_gcb=True)
    elif model_name == "ifblend_blend":
        return IFBlend(16, device=device, use_gcb=True, blend=True)
    elif model_name == "ifblend_nogcb":
        return IFBlend(16, device=device, use_gcb=False)
    else:
        return None


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    for name in ["ifblend_nogcb", "ifblend"]:
        with torch.cuda.device(0):
            net = get_model(name)
            net = net.cuda()
            macs, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True,
                                                     print_per_layer_stat=False, verbose=False)
            print('{} - {:<30}  {:<8}'.format(name, 'Computational complexity: ', macs))
            print('{} - {:<30}  {:<8}'.format(name, 'Number of parameters: ', params))



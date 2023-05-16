import torch


# To save the checkpoint
def save_checkpoint(model, checkpoint_path):
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), checkpoint_path)


# To load the checkpoint
def load_checkpoint(model, checkpoint_path, dp=True):
    # model.module.load_state_dict(torch.load(args.best_model_save_path))
    best_checkpoint = torch.load(checkpoint_path)
    if dp:
        model.module.load_state_dict(best_checkpoint)
    else:
        model.load_state_dict(best_checkpoint)



def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key])
    return dst_str


def count_parameters(model):
    answer = 0
    for p in model.parameters():
        if p.requires_grad:
            answer += p.numel()
            # print(p)
    return answer


class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"
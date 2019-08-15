try:
    import torchsummary
except:
    torchsummary = None

from tabulate import tabulate

BATCH_TEMPLATE = "Epoch [{} / {}], Batch [{} / {}]:"
EPOCH_TEMPLATE = "Epoch [{} / {}]:"
TEST_TEMPLATE = "Epoch [{}] Test:"

def print_iter(curr_epoch=None, epochs=None, batch_i=None, num_batches=None, writer=None, msg=False, **kwargs):
    """
    Formats an iteration. kwargs should be a variable amount of metrics=vals
    Optional Arguments:
        curr_epoch(int): current epoch number (should be in range [0, epochs - 1])
        epochs(int): total number of epochs
        batch_i(int): current batch iteration
        num_batches(int): total number of batches
        writer(SummaryWriter): tensorboardX summary writer object
        msg(bool): if true, doesn't print but returns the message string

    if curr_epoch and epochs is defined, will format end of epoch iteration
    if batch_i and num_batches is also defined, will define a batch iteration
    if curr_epoch is only defined, defines a validation (testing) iteration
    if none of these are defined, defines a single testing iteration
    if writer is not defined, metrics are not saved to tensorboard
    """
    if curr_epoch is not None:
        if batch_i is not None and num_batches is not None and epochs is not None:
            out = BATCH_TEMPLATE.format(curr_epoch + 1, epochs, batch_i, num_batches)
        elif epochs is not None:
            out = EPOCH_TEMPLATE.format(curr_epoch + 1, epochs)
        else:
            out = TEST_TEMPLATE.format(curr_epoch + 1)
    else:
        out = "Testing Results:"

    floatfmt = []
    for metric, val in kwargs.items():
        if "loss" in metric or "recall" in metric or "alarm" in metric or "prec" in metric:
            floatfmt.append(".4f")
        elif "accuracy" in metric or "acc" in metric:
            floatfmt.append(".2f")
        else:
            floatfmt.append(".6f")

        if writer and curr_epoch:
            writer.add_scalar(metric, val, curr_epoch)
        elif writer and batch_i:
            writer.add_scalar(metric, val, batch_i * (curr_epoch + 1))

    out += "\n" + tabulate(kwargs.items(), headers=["Metric", "Value"], tablefmt='github', floatfmt=floatfmt)

    if msg:
        return out
    print(out)

def summary(model, input_dim):
    if torchsummary is None:
        raise(ModuleNotFoundError, "TorchSummary was not found!")
    torchsummary.summary(model, input_dim)

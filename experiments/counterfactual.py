"""
Usage:
    counterfactual [options]

Options:
    --dataset=<n>           Dataset with images and labels eg. mnist, cub200 [default: mnist]
    --thought-vector=<n>    Size of the latent space [default: 32]
    --encoder=<n>           Name of the encoder model [default: simplecnn_7a]
    --decoder=<n>           Name of the decoder model [default: simpledeconv_a]
    --classifier=<n>        Name of the classifier model [default: linear_softmax]
    --epochs=<n>            Number of epochs [default: 10]
    --timestamp=<n>         Timestamp of previously-trained network (new ts if left default) [default: None]
"""

if __name__ == '__main__':
    import docopt
    arguments = docopt.docopt(__doc__)
    # TODO: Instead of calling the old bash script, convert it all to Python
    timestamp = arguments['--timestamp']
    if timestamp == 'None':
        timestamp = int(time.time())
    cmd = "experiments/counterfactual.sh {} {} {} {} {} {} {}".format(
            arguments['--dataset'],
            arguments['--encoder'],
            arguments['--decoder'],
            arguments['--classifier'],
            arguments['--thought-vector'],
            arguments['--epochs'],
            timestamp)
    import subprocess
    print(cmd)
    subprocess.check_call(cmd, shell=True)

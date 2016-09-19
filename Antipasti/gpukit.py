__author__ = "Nasim Rahaman"

__doc__ = """Module to implement multi-GPU methods."""

import numpy as np
import multiprocessing as mp


# Multi-GPU Preprocessing
def multiprep(net, gpuids, splits=2):
    assert len(gpuids) == splits, "Number of splits must match the number of GPUs."

    def run(batch, pos, outq, gpuid):
        # Import theano with GPU-ID. This instance of theano should be bound to the function closure of run() defined
        # below
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(gpuid)
        import theano as th
        reload(th)

        if not hasattr(run, "infer"):
            # Compile theano code for function and cache to save compilation time
            infer = th.function(inputs=[net.x], outputs=net.y, allow_input_downcast=True)
            run.infer = infer
        else:
            # Fetch cached function
            infer = run.infer

        # Run inference function and append to queue
        out = infer(batch)
        outq.put(pos, out)

    def multirun(batch):

        # Split batch by first or second dimension
        splitdim = 0 if batch.shape[0] != 1 else 1

        # Split away
        batchsplits = np.split(batch, splits, axis=splitdim)

        # Make a queue
        q = mp.Queue()

        # Make processes for all workers
        processes = [mp.Process(target=run, args=(batchsplit, procnum, q, gpuid))
                     for procnum, batchsplit, gpuid in zip(range(splits), batchsplits, gpuids)]

        # Start processes
        for process in processes:
            process.start()

        # Exit completed processes
        for process in processes:
            process.join()

        # Get process outputs and sort by position
        results = [q.get() for _ in processes]
        results.sort()
        results = [r[1] for r in results]

        # Concatenate results and return
        outbatch = np.concatenate(tuple(results), axis=splitdim)

        # Return
        return outbatch

    return multirun


if __name__ == "__main__":
    print("Building Antipast")
    import Antipasti.netkit as nk
    import Antipasti.netools as ntl

    prepcnn = nk.convlayer(fmapsin=1, fmapsout=50, kersize=[9, 9, 1], activation=ntl.elu(), Wgc=[-10, 10], bgc=[-10, 10]) + \
          nk.convlayer(fmapsin=50, fmapsout=70, kersize=[9, 9, 1], activation=ntl.elu(), Wgc=[-10, 10], bgc=[-10, 10]) + \
          nk.convlayer(fmapsin=70, fmapsout=100, kersize=[7, 7, 1], activation=ntl.elu(), Wgc=[-10, 10], bgc=[-10, 10]) + \
          nk.convlayer(fmapsin=100, fmapsout=100, kersize=[7, 7, 1], activation=ntl.elu(), Wgc=[-10, 10], bgc=[-10, 10]) + \
          nk.convlayer(fmapsin=100, fmapsout=100, kersize=[5, 5, 1], activation=ntl.elu(), Wgc=[-10, 10], bgc=[-10, 10]) + \
          nk.convlayer(fmapsin=100, fmapsout=100, kersize=[5, 5, 1], activation=ntl.elu(), Wgc=[-10, 10], bgc=[-10, 10])

    prepcnn.feedforward()
    print("Building Multiprep Function Factory")
    mr = multiprep(prepcnn, ['gpu0', 'gpu2'])

    batch = np.random.uniform(size=(1, 20, 1, 512, 512))

    print("Calling MPrep Function")
    by = mr(batch)

    print("Done")

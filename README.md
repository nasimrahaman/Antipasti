# Antipasti
Antipasti is a lightweight toolkit for building and training deep networks with Theano. 

## This project is being rewritten; see [antipasti-tf](https://github.com/nasimrahaman/antipasti-tf).
### Why are we rewriting? 
* Tensorflow to leverage multiple GPUs: we have bought enough GPUs in the mean time to leverage efficient multi-GPU training, which is still somewhat experimental in Theano. We wish to retain the non-functional aspects and the syntax of Antipasti, but with a NetworkX backend for more powerful graph manipulation capabilities.
* More pythonic code (PEP8, streamlined control flow, proper packaging, etc.)

### Why not both Theano and Tensorflow, like Keras? 
* Although Keras remains a beautifully written library, we have found that supporting two libraries simultaneously does neither justice. Our path of choice is to support Keras layers in antipasti-tf without committing to multi-framework support.

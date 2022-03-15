# MLMassSpectrom
ML applications on mass spectrometry data.

* Deep learning (namely, LSTM and attention models) built to use dynamic length inputs (using only peaks) for material classification in `core.NN()`

> Note: the attention model might be broken. I can't remember.

* We mostly use simulated data as of right now, which is generated in `core.SimulationGenerator()` and can be run using `Classify.py`

* We also can visualize the LSTM layers for cool paper visualizations in `core.VisualizeRNNLayers()` and can be used by uncommenting the following lines in `Classify.py`: 

```py
vz = VisualizeRNNLayers()
vz.visualize(self.nn.model, self.nn.gen, self.gen_kwargs['savepath'])
```

> These are commented because it takes longer to run the overall script, hindering prototyping speed.

* Recently, we want to add measured data, this was incorporated in `core.RealDataGenerator()` and can be run using `python Classify_real.py`

* Seeing poor results using only measured data, we aim to incorporate both measured and simulated, this was started in `core.BothSimAndRealGenerator()` and can be run using `python Classify_real_and_sim.py`


## Other files

`synthetic_spectrometry.ipynb` was used for initial development of simulations, which was later abstracted into `core.py`.

`core.py` contains all source code for simulations, neural networks, and visualizations.

`Classify_*.py` are scripts used to run!

`extractor_real_to_sim.ipynb` was made to start working on capturing the class-specific peak distributions, which can then be used to fine-tune the simulations. Note: you can read in simulation information by saving it in the `SimulationGenerator.data_setup` dictionary. This is rather simple but I didn't get to it.

`vizualize_results.ipynb` creates vizualizations of the resulting gridsearch runs. Generally, we see that a higher hidden state gives higher classification accuracy.


## Questions?

Email me at `michael.hopwood@knights.ucf.edu` and if I don't respond then email again or ask Dr. Alexander Mantzaris

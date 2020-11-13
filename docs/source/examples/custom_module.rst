Custom network example
===============================

This page will continue the quick-start by showing how to implement and train your own network.
It is assumed that you already followed the quick-start and set up the example there.


Implementing your own network
-----------------------------------

Prognosais was designed to make designing and training your own network as simple as possible.

Basic example: classification network
*****************************************

The simplest case is that of a 'classification' network, where samples belong to a discrete class (this can be either a single output label or a segmentation).
In this case, only the model itself needs to be implemented.

We start by going to the directory with examples created earlier and creating the virtual environment.
Here we will also create file `my_definitions.py` to contain our custom network

.. code-block:: bash

  cd ~/prognosais
  source env/bin/activate
  cd prognosais_examples/
  touch my_definitions.py

Now open `my_definitions.py` in your favorite editor and past the following into the file and save it:

.. code-block:: python

  from tensorflow.keras.layers import Concatenate, Conv3D, Dense, Flatten, ReLU
  from tensorflow.keras.models import Model
  from PrognosAIs.Model.Architectures.Architecture import ClassificationNetworkArchitecture, NetworkArchitecture

  class SimpleNetwork_3D(ClassificationNetworkArchitecture):
      # We derive this class from the base class of the classification network
      # The class should be name as followed: {arbitrary_name}_2D for a 2D network or {arbitrary_name}_3D for a 3D network
      # In this way Prognosais will automatically chose the appropriate network based on the input dimensions

      def create_model(self):
          # Since we use the ClassificationNetworkArchitecture, we only need to define the function create_model
          # This function should construct the model and return it.

          # The inputs are already automatically defined, we can get them from `self`
          # In this case we assume there is only 1 input (for multiple inputs see more complicated examples later)
          inputs = self.inputs

          # We will now create a very simple model
          conv_1 = Conv3D(filters=4, kernel_size=(2, 2, 2))(inputs)

          relu_1 = ReLU()(conv_1)

          flatten_1 = Flatten()(relu_1)

          dense_1 = Dense(units=256)(flatten_1)

          # Since we use a ClassificationNetworkArchitecture, the outputs are defined already as well
          # In this case by default we get softmax output
          predictions = self.outputs(dense_1)

          # We construct the model and return it

          return Model(inputs=self.inputs, outputs=predictions)

Now we need to edit the `config.yml` file in two places:

1. Under `general` add the following: `custom_definitions_file: my_definitions.py`. This will make prognosais load your file with custom definitions
2. Under `model` change `model_name` to `SimpleNetwork`, this will make sure we use our just defined network.

For the `model_name` parameter you never need to add the _2D or _3D part, prognosais will add this automatically based on the dimensions of the input.

The pipeline can now be run again and this new model will be trained:

.. code-block:: bash

  python example_pipeline.py

Of course the model will perform very poorly since it is quite simple, but of course you can make the model as complex as you want.


Advanced example: multiple inputs/outputs
*****************************************

Creating a network that accepts multiple inputs or outputs is not much more complicated than creating the simple network shown in the previous example.
We will expand the previous simple network to deal with multiple inputs and outputs.
Once again open the `my_definitions.py` file and add the following code:


.. code-block:: python

  class NetworkMultiInputMultiOutput_3D(ClassificationNetworkArchitecture):
      def create_model(self):
          # Once again the inputs are automatically created
          # However, since in our toy example data we only have one input and one output, we need to override the default settings
          self.inputs = self.make_inputs(self.input_shapes, self.input_data_type, squeeze_inputs=False)
          self.outputs = self.make_outputs(self.output_info, self.output_data_type, squeeze_outputs=False)
          # By setting squeeze to False, we ensure that even though we do not have multiple inputs/outputs, the inputs and outputs will
          # still be created as if there were actually multiple inputs and outputs
          # If you are sure that you always have multiple inputs/outputs you can use the self.inputs and self.outputs variables directly
          # Otherwise the above two lines are a safe alternative, making sure your model works regardless of the number of inputs/outputs

          # Now the self.inputs variable is actually a dictionary, where the keys are the different input names and the values the actual inputs
          # In this case apply a different convolutional filter to each input, and then concatenate all the inputs

          input_branches = []
          for i_input in self.inputs.values():
              input_branches.append(Conv3D(filters=4, kernel_size=(2, 2, 2))(i_input))

          # Only concatenate if there is more than 1 input
          if len(input_branches) > 1:
              concat_1 = Concatenate()(input_branches)
          else:
              concat_1 = input_branches[0]

          relu_1 = ReLU()(concat_1)

          flatten_1 = Flatten()(relu_1)

          dense_1 = Dense(units=256)(flatten_1)

          # The output are defined similarly, a dictionary with the keys the names of the outputs
          # Thus we can easily create multiple outputs in the following way:
          predictions = []
          for i_output in self.outputs.values():
              predictions.append(i_output(dense_1))

          # If you want to do different things with your outputs you can of course also do something like:
          # predictions = []
          # predictions.append(Dense(units=5, activation="softmax", name="output_1")
          # predictions.append(Dense(units=15, activation="relu", name="output_2")
          # Make sure that the name matches the output labels as defined in your label file!
          # You can also get the output labels from self.output_info.keys()

          # We construct the model and return it

          return Model(inputs=self.inputs, outputs=predictions)

We now need to change the `config.yml` file to train this new network.
Simply change `model_name` under `model` to `NetworkMultiInputMultiOutput`, this will make sure we use our just defined network.
The model can now be trained:

.. code-block:: bash

  python example_pipeline.py

Of course in this example nothing will change compared to the previous example, since our data only has one input and one output.

Advanced example: non-classification network
*********************************************

In the above examples we have always used a ClassificationNetworkArchitecture, which makes it easier to implement our own network.
However, it is possible to implement any arbitrary network using the more basic NetworkArchitecture, of which we present an example here.

Once again open `my_definitions.py` and add the following:

.. code-block:: python

  class NonClassificationNetwork_3D(NetworkArchitecture):
      # We have now used the NetworkArchitecture as the base class
      # We use the same model as the first basic example, nothing changed here
      def create_model(self):
          # Since we use the ClassificationNetworkArchitecture, we only need to define the function create_model
          # This function should construct the model and return it.

          # We need to load the inputs and outputs, they are not automatically generated in this case
          self.inputs = self.make_inputs(self.input_shapes, self.input_data_type)
          self.outputs = self.make_outputs(self.output_info, self.output_data_type)

          # We will now create a very simple model
          conv_1 = Conv3D(filters=4, kernel_size=(2, 2, 2))(self.inputs)

          relu_1 = ReLU()(conv_1)

          flatten_1 = Flatten()(relu_1)

          dense_1 = Dense(units=256)(flatten_1)

          # Since we use a ClassificationNetworkArchitecture, the outputs are defined already as well
          # In this case by default we get softmax output
          predictions = self.outputs(dense_1)

          # We construct the model and return it

          return Model(inputs=self.inputs, outputs=predictions)

      # However, we now also need to define a make_outputs function, since we do not have default for this for this basic architecture
      @staticmethod
      def make_outputs(
          output_info: dict,
          output_data_type: str,
          activation_type: str = "linear",
          squeeze_outputs: bool = True,
      ) -> dict:
          # The variables output_info and output_date_type are required in any make_outputs function, however apart from that you can
          # create any additional parameters that you want

          # The below code will create a dictionary of outputs (one item for each output) and we create a dense layer with one node and linear activation
          # The dtype is float32 but can be adjusted if required for your problem
          outputs = {}
          for i_output_name in output_info.keys():
              outputs[i_output_name] = Dense(
                  1, name=i_output_name, activation="linear", dtype="float32",
              )

          # To make it easier for cases where there is only one output we will squeeze the output
          # Returning only that output instead of a dict
          if squeeze_outputs and len(outputs) == 1:
              outputs = list(outputs.values())[0]

          return outputs


We cannot train this model as the toy example dataset only has discrete data.
However, this shows how a model can be implemented that has arbitrary outputs.




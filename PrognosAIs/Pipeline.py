import argparse
import copy
import os
import shutil
import sys
import warnings

import PrognosAIs.IO.utils as IO_utils

from PrognosAIs.IO import ConfigLoader
from PrognosAIs.IO import Configs
from PrognosAIs.Model import Evaluators
from PrognosAIs.Model import Trainer
from PrognosAIs.Preprocessing import Preprocessors
from slurmpie import slurmpie


class Pipeline(object):
    def __init__(
        self, config_file: str, preprocess: bool = True, train: bool = True, evaluate: bool = True, samples_folder: str = None,
    ):
        self.path = os.path.dirname(os.path.realpath(__file__))

        self.config_file = config_file
        self.config = ConfigLoader.ConfigLoader(config_file)
        self.input_folder = self.config.get_input_folder()
        self.output_folder = os.path.join(
            self.config.get_output_folder(), self.config.get_specific_output_folder()
        )
        if os.path.exists(self.output_folder):
            temp_i = 0
            original_folder = copy.deepcopy(self.output_folder)
            while os.path.exists(self.output_folder):
                self.output_folder = original_folder + "_" + str(temp_i)
                temp_i += 1
        IO_utils.create_directory(self.output_folder)
        self.config_file = self.config.copy_config(self.output_folder)

        # Paths to the python scripts we need
        self.preprocessor = os.path.join(self.path, "Preprocessing", "Preprocessors.py")
        self.trainer = os.path.join(self.path, "Model", "Trainer.py")
        self.evaluator = os.path.join(self.path, "Model", "Evaluators.py")

        self.preprocess = preprocess
        self.train = train
        self.evaluate = evaluate

        self.preprocessing_config = self.config.get_preprocessings_settings()

        if "saving" in self.preprocessing_config:
            saving_config = Configs.saving_config(self.preprocessing_config["saving"])
            print(saving_config)
        else:
            saving_config = Configs.saving_config(None)

        if samples_folder is None:
            self.samples_folder = os.path.join(self.output_folder, saving_config.out_dir_name)
        else:
            self.samples_folder = samples_folder

    def start_local_pipeline(self):

        if self.preprocess:
            preprocessor = Preprocessors.BatchPreprocessor(
                self.input_folder, self.output_folder, self.preprocessing_config
            )

            self.samples_folder = preprocessor.start()

        if self.train:
            trainer = Trainer.Trainer(self.config, self.samples_folder, self.output_folder)

            self.model_file = trainer.train_model()

        if self.train and self.evaluate:
            evaluator = Evaluators.Evaluator(
                self.model_file, self.samples_folder, self.config_file, self.output_folder,
            )
            evaluator.evaluate()

        elif self.evaluate and not self.train:
            warnings.warn(
                "Cannot evaluate if no model is trained, not doing evaluation", SyntaxWarning
            )

    def start_slurm_pipeline(
        self, preprocess_job: slurmpie.Job, train_job: slurmpie.Job, evaluate_job: slurmpie.Job
    ):
        save_name = self.config.get_save_name()
        self.model_file = os.path.join(self.output_folder, "MODEL", save_name + ".hdf5")
        preprocessing_command = """python -u {command} --config={config_file} --input={input_dir} --output={output_dir}""".format(
            command=self.preprocessor,
            config_file=self.config_file,
            input_dir=self.input_folder,
            output_dir=self.output_folder,
        )

        training_command = "srun python {command} --config={config_file} --input={input_dir} --output={output_dir} --savename={savename}".format(
            command=self.trainer,
            config_file=self.config_file,
            input_dir=self.samples_folder,
            output_dir=self.output_folder,
            savename=save_name,
        )

        evaluation_command = "python -u {command} --config={config_file} --input={input_dir} --output={output_dir} --model={model_file}".format(
            command=self.evaluator,
            config_file=self.config_file,
            input_dir=self.samples_folder,
            output_dir=self.output_folder,
            model_file=self.model_file,
        )

        # Here we make the actual jobs
        prognosAIs_pipeline = slurmpie.Pipeline()

        preprocess_job.script_is_file = False
        if preprocess_job.script is not None:
            preprocess_job.script += preprocessing_command
        else:
            preprocess_job.script = preprocessing_command

        train_job.script_is_file = False
        if train_job.script is not None:
            train_job.script += training_command
        else:
            train_job.script = training_command

        evaluate_job.script_is_file = False
        if evaluate_job.script is not None:
            evaluate_job.script += evaluation_command
        else:
            evaluate_job.script = evaluation_command

        if self.preprocess:
            prognosAIs_pipeline.add(preprocess_job)
        if self.preprocess and self.train:
            prognosAIs_pipeline.add({"afterok": [train_job]}, parent_job=preprocess_job)
        elif self.train:
            prognosAIs_pipeline.add(train_job)

        if self.evaluate:
            if self.train:
                prognosAIs_pipeline.add({"afterok": [evaluate_job]}, parent_job=train_job)
            elif self.preprocess:
                prognosAIs_pipeline.add({"afterok": [evaluate_job]}, parent_job=preprocess_job)
            else:
                prognosAIs_pipeline.add(evaluate_job)

        prognosAIs_pipeline.submit()
        print("Submitted the PrognosAIs pipeline!")

    @classmethod
    def init_from_sys_args(cls, args_in):
        parser = argparse.ArgumentParser(description="Start a PrognosAIs pipeline")
        parser.add_argument(
            "-c",
            "--config",
            required=True,
            help="The location of the PrognosAIs config file",
            metavar="configuration file",
            dest="config",
            type=str,
        )

        parser.add_argument(
            "-p",
            "--preprocess",
            required=False,
            help="Whether the pipeline should include preprocessing",
            dest="preprocess",
            action="store_true",
        )

        parser.add_argument(
            "-t",
            "--train",
            required=False,
            help="Whether the pipeline should include training",
            dest="train",
            action="store_true",
        )

        parser.add_argument(
            "-e",
            "--evaluate",
            required=False,
            help="Whether the pipeline should include evaluation",
            dest="evaluate",
            action="store_true",
        )

        args = parser.parse_args(args_in)

        if not args.preprocess and not args.train and not args.evaluate:
            args.preprocess = True
            args.train = True
            args.evaluate = True

        pipeline = cls(args.config, args.preprocess, args.train, args.evaluate)

        return pipeline


if __name__ == "__main__":
    pipeline = Pipeline.init_from_sys_args(sys.argv[1:])
    pipeline.start_local_pipeline()

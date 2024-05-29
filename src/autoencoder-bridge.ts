import AE from "./autoencoder";
import { INeuralNetworkData, INeuralNetworkDatum, INeuralNetworkTrainOptions } from "./neural-network";
import { NeuralNetworkGPU } from "./neural-network-gpu";
import { INeuralNetworkState } from "./neural-network-types";

function notYetInitializedError() {
  return new Error("The network must be trained before running.");
}

export interface IAEBOptions<InputType extends INeuralNetworkData, OutputType extends INeuralNetworkData> {
  binaryThresh: number;
  inputAE: AE<InputType, Float32Array>;
  outputAE: AE<OutputType, Float32Array>;
}

/**
 * An autoencoder bridge (AEB) is a set of 3 autoencoders designed to serve as a bridge between two dissimilar data formats.
 *
 * Two existing autoencoders are used to train a third autoencoder which handles translations between the two encodings.
 */
export class AEBridge<InputType extends INeuralNetworkData, OutputType extends INeuralNetworkData> {
  #binaryThresh: number;
  #inputAE: AE<InputType, Float32Array>;
  #outputAE: AE<OutputType, Float32Array>;
  #mergeAE?: NeuralNetworkGPU<Float32Array, Float32Array>;

  mergeTrainingData(inputData: InputType[], outputData: OutputType[]): Array<INeuralNetworkDatum<Partial<InputType>, Partial<OutputType>>> {
    return inputData.map(
      (input, index) => ({ input, output: outputData[index] })
    );
  }

  constructor(options?: Partial<IAEBOptions<InputType, OutputType>>) {
    options ??= {};

    if (!options.inputAE) throw notYetInitializedError();
    if (!options.outputAE) throw notYetInitializedError();

    this.#binaryThresh = options.binaryThresh ?? 0.5;

    this.#inputAE = options.inputAE;
    this.#outputAE = options.outputAE;
  }

  forward(input: InputType) {
    if (!this.#inputAE || !this.#mergeAE || !this.#outputAE) throw notYetInitializedError();

    return this.#outputAE.decode(this.#mergeAE.run(this.#inputAE.encode(input)));
  }

  backward(input: OutputType) {
    if (!this.#inputAE || !this.#mergeAE || !this.#outputAE) throw notYetInitializedError();

    return this.#inputAE.decode(this.#mergeAE.run(this.#outputAE.encode(input)));
  }

  train(data: Array<INeuralNetworkDatum<Partial<InputType>, Partial<OutputType>>>, options?: Partial<INeuralNetworkTrainOptions> | undefined): INeuralNetworkState {
    const trainingData: Array<INeuralNetworkDatum<Partial<Float32Array>, Partial<Float32Array>>> = data.map(
      value => ({ input: this.#inputAE.encode(value.input as InputType), output: this.#outputAE.encode(value.output as OutputType) })
    );

    const inputSize = trainingData[0].input.length ?? 1;
    const outputSize = trainingData[0].output.length ?? 1;
    const hiddenSize = Math.max(1, Math.round(outputSize * 0.6));

    const mergeAE = new NeuralNetworkGPU<Float32Array, Float32Array>(
      {
        binaryThresh: this.#binaryThresh,
        inputSize,
        hiddenLayers: [ inputSize, hiddenSize ],
        outputSize
      }
    );

    this.#mergeAE = mergeAE;

    return mergeAE.train(trainingData, options);
  }
}

export default AEBridge;

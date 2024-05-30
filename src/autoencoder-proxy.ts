import AE from "./autoencoder";
import UntrainedNeuralNetworkError from "./errors/untrained-neural-network-error";
import { INeuralNetworkData, INeuralNetworkDatum, INeuralNetworkTrainOptions } from "./neural-network";
import { NeuralNetworkGPU } from "./neural-network-gpu";
import { INeuralNetworkState } from "./neural-network-types";

export interface IAEProxyOptions<InputType extends INeuralNetworkData, OutputType extends INeuralNetworkData> {
  binaryThresh: number;
  input: AE<InputType, Float32Array>;
  output: AE<OutputType, Float32Array>;
}

/**
 * An autoencoder bridge (AEB) is a set of 3 autoencoders designed to serve as a bridge between two dissimilar data formats.
 *
 * Two existing autoencoders are used to train a third autoencoder which handles translations between the two encodings.
 */
export class AEProxy<InputType extends INeuralNetworkData, OutputType extends INeuralNetworkData> {
  #binaryThresh: number;
  #inputAE: AE<InputType, Float32Array>;
  #outputAE: AE<OutputType, Float32Array>;
  #proxyAE?: NeuralNetworkGPU<Float32Array, Float32Array>;

  proxyTrainingData(inputData: InputType[], outputData: OutputType[]): Array<INeuralNetworkDatum<Partial<InputType>, Partial<OutputType>>> {
    return inputData.map(
      (input, index) => ({ input, output: outputData[index] })
    );
  }

  constructor(options?: Partial<IAEProxyOptions<InputType, OutputType>>) {
    options ??= {};

    if (!options.input) throw new UntrainedNeuralNetworkError();
    if (!options.output) throw new UntrainedNeuralNetworkError();

    this.#binaryThresh = options.binaryThresh ?? 0.5;

    this.#inputAE = options.input;
    this.#outputAE = options.output;
  }

  forward(input: InputType) {
    if (!this.#inputAE || !this.#proxyAE || !this.#outputAE) throw new UntrainedNeuralNetworkError();

    return this.#outputAE.decode(this.#proxyAE.run(this.#inputAE.encode(input)));
  }

  reverse(output: OutputType) {
    if (!this.#inputAE || !this.#proxyAE || !this.#outputAE) throw new UntrainedNeuralNetworkError();

    return this.#inputAE.decode(this.#proxyAE.run(this.#outputAE.encode(output)));
  }

  train(data: Array<INeuralNetworkDatum<Partial<InputType>, Partial<OutputType>>>, options?: Partial<INeuralNetworkTrainOptions> | undefined): INeuralNetworkState {
    const trainingData: Array<INeuralNetworkDatum<Partial<Float32Array>, Partial<Float32Array>>> = data.map(
      value => ({ input: this.#inputAE.encode(value.input as InputType), output: this.#outputAE.encode(value.output as OutputType) })
    );

    const inputSize = trainingData[0].input.length ?? 1;
    const outputSize = trainingData[0].output.length ?? 1;
    const hiddenSize = Math.max(1, Math.round(outputSize * 0.6));

    const proxyAE = new NeuralNetworkGPU<Float32Array, Float32Array>(
      {
        binaryThresh: this.#binaryThresh,
        inputSize,
        hiddenLayers: [ inputSize, hiddenSize ],
        outputSize
      }
    );

    this.#proxyAE = proxyAE;

    return proxyAE.train(trainingData, options);
  }
}

export default AEProxy;

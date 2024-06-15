import { IGPUKernelSettings, IKernelFunctionThis, IKernelRunShortcut, KernelOutput, Texture } from "gpu.js";
import { INeuralNetworkData, INeuralNetworkDatum, INeuralNetworkTrainOptions } from "./neural-network";
import { INeuralNetworkGPUOptions, NeuralNetworkGPU } from "./neural-network-gpu";
import { release } from "./utilities/kernel";
import { INeuralNetworkState } from "./neural-network-types";

const DEFAULT_LEARNING_RATE = 0.1;

type RandomVectorFunction = (
  this: IKernelFunctionThis,
  learningRate: number
) => number;

type LossFunction = (
  this: IKernelFunctionThis,
  actual: number[],
  expected: number[],
  errors: number[],
  learningRate: number,
  layerSize: number
) => number;

function randomDeltas(
  this: IKernelFunctionThis,
  learningRate: number
) {
  return Math.random() * learningRate;
}

function randomErrors(
  this: IKernelFunctionThis,
  learningRate: number
) {
  return Math.random() * learningRate;
}

export class ELM<InputType extends INeuralNetworkData, OutputType extends INeuralNetworkData> extends NeuralNetworkGPU<InputType, OutputType> {
  randomDeltas: IKernelRunShortcut[] = [];
  randomErrors: IKernelRunShortcut[] = [];

  _learningRate: number = DEFAULT_LEARNING_RATE;

  public get learningRate() {
    return this._learningRate;
  }

  public set learningRate(
    value: number
  ) {
    this._learningRate = value;
  }

  constructor(
    options?: Partial<INeuralNetworkGPUOptions>
  ) {
    super(options);

    this._resize();
  }

  calculateDeltas = (target: KernelOutput): void => {
    const learningRate = this.learningRate;
    for (let layer = this.outputLayer; layer > 0; layer--) {
      release(this.deltas[layer]);
      release(this.errors[layer]);

      let output;
      if (layer === this.outputLayer) {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-expect-error
        output = this.backwardPropagate[layer](this.outputs[layer], target);
      } else {
        const layerSize = (this.biases[layer] as number[]).length;
        const errors = this.randomErrors[layer](learningRate);
        const result = this.randomDeltas[layer](learningRate);
        output = {
          errors,
          result
        }
      }
      this.deltas[layer] = output.result;
      this.errors[layer] = output.error;
    }
  };

  train(data: INeuralNetworkDatum<Partial<InputType>, Partial<OutputType>>[], options?: Partial<INeuralNetworkTrainOptions>): INeuralNetworkState {
    const learningRate = options?.learningRate ?? 0.1;

    this.learningRate = learningRate;

    return super.train(data, options);
  }

  private _resize() {
    for (
      let layer = 0;
      layer < this.sizes.length - 1;
      layer++
    ) {
      const layerSize = this.sizes[layer];
      const options: IGPUKernelSettings = {
        output: [ layerSize ],
        pipeline: true
      };
      this.randomErrors[layer] = this.gpu.createKernel(randomErrors, options);
      this.randomDeltas[layer] = this.gpu.createKernel(randomDeltas, options);
    }
  }
}

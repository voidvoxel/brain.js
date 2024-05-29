export class UntrainedNeuralNetworkError extends Error {
  constructor () {
    super("The neural network must be trained before running.");
  }
}

export default UntrainedNeuralNetworkError;

import AE from "./autoencoder";
import AEProxy from "./autoencoder-proxy";

const trainingDataXOR = [
  [0, 0, 0],
  [0, 1, 1],
  [1, 0, 1],
  [1, 1, 0]
];

const errorThresh = 0.007;

const xornet = new AE<number[], Float32Array>(
  {
    decodedSize: 3,
    hiddenLayers: [ 3, 2 ]
  }
);

xornet.train(
  trainingDataXOR, {
    iterations: 500000,
    errorThresh
  }
);

const trainingDataNXOR = [
  [0, 0, 1],
  [0, 1, 0],
  [1, 0, 0],
  [1, 1, 1]
];

const nxornet = new AE<number[], Float32Array>(
  {
    decodedSize: 3,
    hiddenLayers: [ 3, 2 ]
  }
);

nxornet.train(
  trainingDataNXOR, {
    iterations: 500000,
    errorThresh
  }
);

const aeProxy = new AEProxy({
  input: xornet,
  output: nxornet
});

const trainingDataProxy = aeProxy.proxyTrainingData(trainingDataXOR, trainingDataNXOR);

const result = aeProxy.train(
  trainingDataProxy, {
    iterations: 25000,
    errorThresh
  }
);

test(
  "autoencoder proxy accuracy",
  async () => {
    expect(result.error).toBeLessThanOrEqual(errorThresh);
  }
);

test(
  "forward proxy",
  async () => {
    function xor(...args: number[]) {
      return Math.round(aeProxy.forward(args)[2]);
    }

    const run1 = xor(0, 0, 0);
    const run2 = xor(0, 1, 1);
    const run3 = xor(1, 0, 1);
    const run4 = xor(1, 1, 0);

    expect(run1).toBe(1);
    expect(run2).toBe(0);
    expect(run3).toBe(0);
    expect(run4).toBe(1);
  }
);

test(
  "reverse proxy",
  async () => {
    function xor(...args: number[]) {
      return Math.round(aeProxy.reverse(args)[2]);
    }

    const run1 = xor(0, 0, 1);
    const run2 = xor(0, 1, 0);
    const run3 = xor(1, 0, 0);
    const run4 = xor(1, 1, 1);

    expect(run1).toBe(0);
    expect(run2).toBe(1);
    expect(run3).toBe(1);
    expect(run4).toBe(0);
  }
);

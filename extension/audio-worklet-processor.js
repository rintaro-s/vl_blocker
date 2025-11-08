class CaptureProcessor extends AudioWorkletProcessor {
	static get parameterDescriptors() {
		return [];
	}

	constructor() {
		super();
		this.channelCount = 1;
	}

	process(inputs) {
		const input = inputs[0];
		if (!input || input.length === 0) {
			return true;
		}

		const channelData = input[0];
		if (!channelData || channelData.length === 0) {
			return true;
		}

		const copy = new Float32Array(channelData.length);
		copy.set(channelData);
		this.port.postMessage(copy, [copy.buffer]);
		return true;
	}
}

registerProcessor('capture-processor', CaptureProcessor);

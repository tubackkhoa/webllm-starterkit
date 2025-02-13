import * as webllm from '@mlc-ai/web-llm';
import { marked } from 'marked';

const selectedModel = 'Llama-3.1-8B-Instruct-q4f32_1-MLC';

const appConfig = {
  model_list: [
    {
      model: 'https://huggingface.co/mlc-ai/' + selectedModel, // a base model
      model_id: selectedModel,
      model_lib:
        webllm.modelLibURLPrefix +
        webllm.modelVersion +
        '/' +
        selectedModel.replace(/\./g, '_').replace(/MLC$/, '') +
        'ctx4k_cs1k-webgpu.wasm',
      overrides: {
        context_window_size: 2048
      }
    }
  ]
};

let engine: webllm.MLCEngineInterface;

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error('Cannot find label ' + id);
  }
  label.innerText = text;
}

/**
 * Chat completion (OpenAI style) with streaming, where delta is sent while generating response.
 */
async function mainStreaming() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel('init-label', report.text);
  };

  engine = await webllm.CreateMLCEngine(selectedModel, {
    appConfig,
    initProgressCallback: initProgressCallback
  });
}

async function chat(msg: string) {
  const request: webllm.ChatCompletionRequest = {
    stream: true,
    stream_options: { include_usage: false },

    messages: [
      {
        role: 'system',
        content: 'You are a helpful, respectful and honest assistant.'
      },
      { role: 'user', content: msg }
    ],
    logprobs: true,
    top_logprobs: 2,
    max_tokens: 256
  };

  const asyncChunkGenerator = await engine.chat.completions.create(request);
  let message = '';
  for await (const chunk of asyncChunkGenerator) {
    let sub = chunk.choices[0]?.delta?.content;
    if (sub) {
      message += sub;
      document.getElementById('generate-label')!.innerHTML = await marked.parse(
        message
      );
    }
  }
}

document.getElementById('chat')?.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') {
    // @ts-ignore
    chat(ev.target.value);
  }
});

// Run one of the function below
// mainNonStreaming();
mainStreaming();

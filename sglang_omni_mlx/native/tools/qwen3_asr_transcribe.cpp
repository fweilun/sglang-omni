// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native runtime and prints one JSON line each,
// for parity checks against the Python reference server. --windowed replays
// each file as a realtime live feed instead.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "common/audio.h"
#include "common/tokenizer.h"
#include "nlohmann/json.hpp"
#include "qwen3_asr/qwen3_asr_model.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_asr/realtime.h"

namespace {

// Voxt's live feed appends 0.1 s at a time; the rest follow realtime.cpp.
constexpr long kAppendSampleCount = qwen3_asr::kSampleRate / 10;
constexpr int kPrefixAfterRefreshCount = 2;
constexpr int kPrefixRollbackTokenCount = 5;
constexpr float kSilentPeak = 1e-3f;

// Feeds samples to one realtime session as the server handles it with its
// default settings: previews on the decode cadence reuse their segment's
// StreamingEncoder, and finals at each max segment cut and at the end encode
// anew. Previews land as if each decode ended within the interval. Prints one
// JSON line per decode; returns the finals joined.
std::string ReplayLiveFeed(const qwen3_asr::Qwen3ASRTranscriber &transcriber,
                           const std::vector<float> &samples,
                           const qwen3_asr::TranscriptionOptions &options,
                           const std::atomic<bool> &cancel) {
  const qwen3_asr::RealtimeSettings settings = qwen3_asr::MakeRealtimeSettings(
      /*decode_interval_ms=*/1000, /*first_decode_ms=*/100,
      /*max_segment_seconds=*/30.0);
  struct Segment {
    long start_sample = 0;
    long next_refresh_sample = 0;
    int decode_count = 0;
    std::optional<std::string> language;
    std::string transcript;
    std::unique_ptr<qwen3_asr::StreamingEncoder> encoder;
  };
  const auto new_segment = [&](long start_sample) {
    Segment segment;
    segment.start_sample = start_sample;
    segment.next_refresh_sample = start_sample + settings.first_decode_samples;
    segment.language = options.language;
    segment.encoder = std::make_unique<qwen3_asr::StreamingEncoder>(
        transcriber, options.layout);
    return segment;
  };
  const auto is_silent = [&](const Segment &segment, long end_sample) {
    return qwen3_asr::PeakIsSilent(samples.data() + segment.start_sample,
                                   end_sample - segment.start_sample,
                                   kSilentPeak);
  };
  const auto decode = [&](Segment &segment, long end_sample, bool is_final) {
    qwen3_asr::TranscriptionOptions decode_options = options;
    decode_options.language = segment.language;
    if (segment.decode_count >= kPrefixAfterRefreshCount &&
        !segment.transcript.empty() && segment.language.has_value()) {
      std::tie(decode_options.prefix_token_ids, decode_options.prefix_text) =
          transcriber.RetainedPrefix(segment.transcript,
                                     kPrefixRollbackTokenCount);
    } else {
    }
    segment.decode_count += 1;
    const qwen3_asr::TranscriptionResult result = transcriber.Transcribe(
        std::vector<float>(samples.begin() + segment.start_sample,
                           samples.begin() + end_sample),
        decode_options, cancel, is_final ? nullptr : segment.encoder.get());
    if (result.language.has_value() && !result.language->empty()) {
      segment.language = result.language;
    } else {
    }
    segment.transcript = result.text;
    std::cout << nlohmann::ordered_json(
                     {{"is_final", is_final},
                      {"segment_s",
                       static_cast<double>(end_sample - segment.start_sample) /
                           qwen3_asr::kSampleRate},
                      {"melenc_ms", result.melenc_ms},
                      {"prefill_ms", result.prefill_ms},
                      {"decode_ms", result.decode_ms},
                      {"gen_tokens", result.generated_token_count},
                      {"text", result.text}})
                     .dump()
              << std::endl;
  };
  std::vector<std::string> finals;
  const auto finalize = [&](Segment &segment, long end_sample) {
    if (is_silent(segment, end_sample)) {
      finals.emplace_back();
    } else {
      decode(segment, end_sample, true);
      finals.push_back(segment.transcript);
    }
  };

  const long total_sample_count = static_cast<long>(samples.size());
  Segment segment = new_segment(0);
  long end_sample = 0;
  while (end_sample < total_sample_count) {
    end_sample = std::min(total_sample_count, end_sample + kAppendSampleCount);
    if (end_sample - segment.start_sample >= settings.max_segment_samples) {
      const long cut_sample =
          segment.start_sample + settings.max_segment_samples;
      finalize(segment, cut_sample);
      segment = new_segment(cut_sample);
    } else {
    }
    if (end_sample >= segment.next_refresh_sample) {
      const bool silent = is_silent(segment, end_sample);
      // Note (Jiaxin Deng): leading silence keeps the early first decode for
      // the first audible audio.
      if (!(silent && segment.decode_count == 0)) {
        segment.next_refresh_sample =
            end_sample + settings.decode_interval_samples;
      } else {
      }
      if (!silent) {
        decode(segment, end_sample, false);
      } else {
      }
    } else {
    }
  }
  if (end_sample > segment.start_sample) {
    finalize(segment, end_sample);
  } else {
  }
  return qwen3_asr::JoinTranscriptParts(finals);
}

} // namespace

int main(int argc, char **argv) {
  using qwen3_asr::AudioLayout;
  std::string model_path;
  std::string encode_text;
  std::string dump_mel_filters;
  std::string encode_lines;
  bool encode = false;
  bool profile = false;
  bool windowed = false;
  qwen3_asr::TranscriptionOptions options;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument(argument + " needs a value");
      return argv[++i];
    };
    if (argument == "--model-path") {
      model_path = value();
    } else if (argument == "--layout") {
      options.layout = value() == "voxt_swift" ? AudioLayout::kVoxtSwift
                                               : AudioLayout::kReference;
    } else if (argument == "--language") {
      options.language = qwen3_asr::NormalizeLanguage(value());
    } else if (argument == "--stop-at-end-of-text") {
      options.stop_at_end_of_text = true;
    } else if (argument == "--stop-on-token-loop") {
      options.stop_on_token_loop = true;
    } else if (argument == "--max-new-tokens") {
      options.max_new_tokens = std::stoi(value());
    } else if (argument == "--encode") {
      encode = true;
      encode_text = value();
    } else if (argument == "--encode-lines") {
      encode_lines = value();
    } else if (argument == "--profile") {
      profile = true;
    } else if (argument == "--windowed") {
      windowed = true;
    } else if (argument == "--dump-mel-filters") {
      dump_mel_filters = value();
    } else {
      files.push_back(argument);
    }
  }
  if (!dump_mel_filters.empty()) {
    std::ofstream out(dump_mel_filters, std::ios::binary);
    const auto &filters = qwen3_asr::MelFilterBank();
    const auto &window = qwen3_asr::PeriodicHannWindow();
    out.write(reinterpret_cast<const char *>(filters.data()),
              filters.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(window.data()),
              window.size() * sizeof(float));
    return 0;
  } else if (!encode_lines.empty()) {
    const qwen3_asr::Tokenizer tokenizer(model_path);
    std::ifstream lines(encode_lines);
    std::string line;
    while (std::getline(lines, line)) {
      const std::string text = nlohmann::json::parse(line).get<std::string>();
      const std::vector<int> ids = tokenizer.Encode(text);
      std::cout << nlohmann::json(
                       {{"ids", ids},
                        {"decoded", tokenizer.Decode(ids, false)},
                        {"decoded_skip", tokenizer.Decode(ids, true)}})
                       .dump()
                << "\n";
    }
    return 0;
  } else if (profile) {
    namespace mx = mlx::core;
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const qwen3_asr::Qwen3ASR model(model_path);
    const qwen3_asr::Tokenizer tokenizer(model_path);
    std::ifstream stream(files.at(0), std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    const std::vector<float> samples = qwen3_asr::DecodeWav(bytes.str());
    for (int round = 0; round < 3; ++round) {
      const auto t0 = Clock::now();
      mx::array mel = qwen3_asr::LogMel(samples, AudioLayout::kVoxtSwift);
      mx::eval(mel);
      const auto t1 = Clock::now();
      mx::array encoded = model.EncodeAudio(mel, AudioLayout::kVoxtSwift);
      mx::eval(encoded);
      const auto t2 = Clock::now();
      std::string prompt =
          "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>";
      for (int i = 0;
           i < qwen3_asr::TokenCount(mel.shape(-1), AudioLayout::kVoxtSwift);
           ++i)
        prompt += "<|audio_pad|>";
      prompt += "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage "
                "English<asr_text>";
      const std::vector<int> ids = tokenizer.Encode(prompt);
      const mx::array input(ids.data(), {1, static_cast<int>(ids.size())},
                            mx::int32);
      std::vector<qwen3_asr::KVCache> caches = model.NewCaches();
      mx::array token =
          mx::argmax(model.Decode(model.EmbedTokens(input), caches));
      mx::eval(token);
      const auto t3 = Clock::now();
      constexpr int steps = 100;
      for (int i = 0; i < steps; ++i) {
        token = mx::argmax(model.Decode(
            model.EmbedTokens(mx::reshape(token, {1, 1})), caches));
        mx::async_eval({token});
      }
      mx::eval(token);
      const auto t4 = Clock::now();
      std::cout << "samples " << samples.size() / 16000.0 << "s mel "
                << ms(t0, t1) << "ms enc " << ms(t1, t2) << "ms prefill("
                << ids.size() << ") " << ms(t2, t3) << "ms decode "
                << ms(t3, t4) / steps << "ms/token\n";
    }
    return 0;
  } else if (encode) {
    const qwen3_asr::Tokenizer tokenizer(model_path);
    std::cout << nlohmann::json(tokenizer.Encode(encode_text)).dump() << "\n";
    return 0;
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  const qwen3_asr::Qwen3ASRTranscriber transcriber(model_path);
  std::cerr << "loaded in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             load_started)
                   .count()
            << " s\n";
  const std::atomic<bool> cancel(false);
  for (const auto &file : files) {
    std::ifstream stream(file, std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    const auto started = std::chrono::steady_clock::now();
    const std::vector<float> samples = qwen3_asr::DecodeWav(bytes.str());
    nlohmann::json line;
    if (windowed) {
      line = {{"file", file},
              {"text", ReplayLiveFeed(transcriber, samples, options, cancel)}};
    } else {
      const auto result = transcriber.Transcribe(samples, options, cancel);
      line = {
          {"file", file},
          {"text", result.text},
          {"language", result.language.has_value()
                           ? nlohmann::json(*result.language)
                           : nlohmann::json()},
          {"generated_token_count", result.generated_token_count},
          {"finish_reason", qwen3_asr::FinishReasonName(result.finish_reason)},
      };
    }
    line["seconds"] = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - started)
                          .count();
    std::cout << line.dump() << std::endl;
    // Note (Jiaxin Deng): as the server does between requests; buffers of one
    // clip's sizes rarely fit the next, so the cache would grow without end.
    mlx::core::clear_cache();
  }
  return 0;
}

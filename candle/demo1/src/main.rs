use candle_examples::whisper;
use candle_core::Device;
use anyhow::Result;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-audio.wav>", args[0]);
        std::process::exit(1);
    }

    let audio_path = PathBuf::from(&args[1]);

    // 模型路径可以从 HuggingFace 上下载，例如 whisper-tiny
    let model_path = PathBuf::from("models/whisper-tiny");

    // 使用 CPU 设备（你也可以尝试 GPU）
    let device = Device::Cpu;

    // 加载模型
    let mut whisper = whisper::Whisper::load(&model_path, &device)?;

    // 加载音频并转为特征
    let input = whisper::Whisper::load_audio(&audio_path)?;
    let mel = whisper::Whisper::audio_to_mel(&input)?;

    // 执行识别
    let result = whisper.decode(&mel)?;

    println!("Transcription:\n{}", result.text);

    Ok(())
}

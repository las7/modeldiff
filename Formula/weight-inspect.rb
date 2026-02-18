class WeightInspect < Formula
  desc "Tool for inspecting and diffing model weights (GGUF, SafeTensors)"
  homepage "https://github.com/las7/weight-inspect"
  url "https://github.com/las7/weight-inspect.git"
  version "0.1.0"
  license "MIT"

  depends_on "rust" => :build

  def install
    system "cargo", "build", "--release"
    bin.install "target/release/weight-inspect"
  end

  test do
    system "#{bin}/weight-inspect", "--version"
  end
end

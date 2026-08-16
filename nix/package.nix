{
  lib,
  rustPlatform,
  pkg-config,
  cmake,
  llvmPackages,
  makeWrapper,
  makeFontsConf,
  wayland,
  libxkbcommon,
  vulkan-loader,
  libGL,
  libx11,
  libxcursor,
  libxi,
  libxrandr,
  fontconfig,
  dejavu_fonts,
}:

let
  runtimeLibs = [
    vulkan-loader
    libGL
    wayland
    libxkbcommon
    libx11
    libxcursor
    libxi
    libxrandr
  ];
in
rustPlatform.buildRustPackage {
  pname = "keyb";
  version = "0.1.0";

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../crates
    ];
  };

  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [
    pkg-config
    cmake
    llvmPackages.clang
    rustPlatform.bindgenHook
    makeWrapper
  ];

  buildInputs = runtimeLibs ++ [ fontconfig ];

  env.LIBCLANG_PATH = "${lib.getLib llvmPackages.libclang}/lib";

  # cmake is only for llama-cpp-sys-2's build.rs, not a CMake project.
  dontUseCmakeConfigure = true;

  postInstall = ''
    install -Dm644 ${./keyb.desktop} $out/share/applications/keyb.desktop
    wrapProgram $out/bin/keyb \
      --prefix PATH : ${lib.makeBinPath [ fontconfig ]} \
      --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath runtimeLibs} \
      --set FONTCONFIG_FILE ${makeFontsConf { fontDirectories = [ dejavu_fonts ]; }}
  '';

  meta = {
    description = "Mouse-driven zooming on-screen keyboard";
    license = with lib.licenses; [
      mit
      asl20
    ];
    platforms = lib.platforms.linux;
    mainProgram = "keyb";
  };
}

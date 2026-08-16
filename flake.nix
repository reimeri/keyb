{
  description = "Mouse-driven zooming on-screen keyboard";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forEachSystem =
        f: nixpkgs.lib.genAttrs systems (system: f system nixpkgs.legacyPackages.${system});
    in
    {
      packages = forEachSystem (
        _system: pkgs: rec {
          keyb = pkgs.callPackage ./nix/package.nix { };
          default = keyb;
        }
      );

      apps = forEachSystem (system: _pkgs: {
        default = {
          type = "app";
          program = "${self.packages.${system}.keyb}/bin/keyb";
        };
      });

      devShells = forEachSystem (
        system: pkgs:
        let
          keyb = self.packages.${system}.keyb;
          runtimeLibs = with pkgs; [
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
        {
          default = pkgs.mkShell {
            inputsFrom = [ keyb ];
            packages = with pkgs; [
              rust-analyzer
              rustfmt
              clippy
            ];
            env = {
              LIBCLANG_PATH = "${pkgs.lib.getLib pkgs.llvmPackages.libclang}/lib";
              RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
              FONTCONFIG_FILE = pkgs.makeFontsConf {
                fontDirectories = [ pkgs.dejavu_fonts ];
              };
            };
          };
        }
      );
    };
}

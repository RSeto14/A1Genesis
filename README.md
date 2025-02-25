# A1用のDemo開発

## 参考

- [Genesisの公式ドキュメント](https://genesis-world.readthedocs.io/en/latest/index.html)
- [GenesisのGitHub](https://github.com/Genesis-Embodied-AI/Genesis)
- [誰かのQiita](https://qiita.com/hEnka/items/cc5fd872eb0bf7cd3abc)

## 0. 注意事項

- GPUを使うことが前提
- Linux環境でないと描画ができない(Windowsでは学習のみできる)
- Anacondaはインストール済みであること

## 1. 環境構築

```bash
conda create -n genesis-env python=3.11
conda activate genesis-env
```

## 2. ライブラリインストール

### 2.1. PyTorch

公式ページからインストール

<url>https://pytorch.org/</url>
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.2. その他のライブラリ

```bash
pip install genesis-world
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .
pip install tensorboard
pip install pytz
pip install PyYAML
pip install open3d
```

## 3. Unitree A1 (Demo作成)

- [Scripts/CPG-RL](Scripts/Unitree-A1/CPG-RL) ：[CPG-RL](https://ieeexplore.ieee.org/document/9932888) のフレームワークを用いた平地での歩行学習

- [Scripts/Res4CPG](Scripts/Unitree-A1/Res4CPG) ：残差学習を用いた不整地での歩行学習


## 4. その他

### xmlファイルの挿入

```python
self.robot = self.scene.add_entity(
       gs.morphs.MJCF(
              file=f"{parent_dir_2}/Description/A1/a1.xml",
              # pos=self.base_init_pos.cpu().numpy(),
              # quat=self.base_init_quat.cpu().numpy(),
       ),
       )
```

actuatorをgeneralに変更する必要がある

```xml
  <actuator>
    <general class="abduction" name="FR_hip" joint="FR_hip_joint"/>
    <general class="hip" name="FR_thigh" joint="FR_thigh_joint"/>
    <general class="knee" name="FR_calf" joint="FR_calf_joint"/>
    <general class="abduction" name="FL_hip" joint="FL_hip_joint"/>
    <general class="hip" name="FL_thigh" joint="FL_thigh_joint"/>
    <general class="knee" name="FL_calf" joint="FL_calf_joint"/>
    <general class="abduction" name="RR_hip" joint="RR_hip_joint"/>
    <general class="hip" name="RR_thigh" joint="RR_thigh_joint"/>
    <general class="knee" name="RR_calf" joint="RR_calf_joint"/>
    <general class="abduction" name="RL_hip" joint="RL_hip_joint"/>
    <general class="hip" name="RL_thigh" joint="RL_thigh_joint"/>
    <general class="knee" name="RL_calf" joint="RL_calf_joint"/>
  </actuator>
```

### 地形

凹凸のある地形を生成する[公式ページ](https://genesis-world.readthedocs.io/en/latest/api_reference/options/morph/file_morph/terrain.html#genesis.options.morphs.Terrain)

```python
self.terrain = self.scene.add_entity(gs.morphs.Terrain(n_subterrains=(2,2),subterrain_types=[["fractal_terrain","fractal_terrain"],["fractal_terrain","fractal_terrain"]],subterrain_size=(10,10),horizontal_scale=0.25,vertical_scale=0.005,visualization=True))
```


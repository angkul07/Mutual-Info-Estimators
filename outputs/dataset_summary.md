# Dataset Summary — `place_the_bread`

## Source
- **Repository**: angkul07/abc-ego
- **Task**: `place_the_bread`
- **Total episodes**: 1362
- **Format**: MCAP (ROS2 bag)
- **Inspected from**: `cache/place_the_bread/episode_00242bb8-86f6-4b90-a381-15203ec1a501/episode.mcap`

## Episode IDs (sample)

| # | Episode ID |
|---|------------|
| 1 | `00242bb8-86f6-4b90-a381-15203ec1a501` |
| 2 | `002dea7e-bb58-4a39-a95e-50abc6004068` |
| 3 | `0145b7a5-eeaf-4f25-b2b1-b682b32fcf0f` |
| 4 | `017d7767-8296-42fd-a444-aa4f052c4505` |
| 5 | `01d78a8a-0492-447b-956c-f620071982f6` |
| 6 | `01d83ae0-4f3a-4f06-92a2-56e08d6ec2ad` |
| 7 | `021c9083-3167-4977-85cb-55fb7e9490d7` |
| 8 | `022e6b6b-e5c4-411f-8e71-fe69c19b72e3` |
| 9 | `025a585c-1997-492a-b76d-a91ff979c6f2` |
| 10 | `02600b80-f7e1-4ebc-914a-f62339b0f918` |
| 11 | `0263100e-dc79-4128-a7f3-b4c643b59d4a` |
| 12 | `029f98aa-f967-41a5-b1e1-4f0fac1bedd4` |
| 13 | `02d4401b-dbf9-4ffd-81f4-53ec4462cc82` |
| 14 | `038b507e-4172-44a5-8abc-d7bed81ef8ad` |
| 15 | `038b65ba-c5b0-48ac-893c-66c68de9f453` |
| 16 | `045349d3-d7ad-4b9d-b6d4-1045c2ca53a9` |
| 17 | `046eff2b-81eb-4b63-93d1-83745bf796bd` |
| 18 | `0475224c-e862-4787-b6f3-f32f36e231e5` |
| 19 | `04a46440-4dd6-4df2-ac08-e2e5f11e9191` |
| 20 | `04b04e98-9cbd-453e-9627-62cfe0faf6f1` |
| … | *(+1342 more)* |

## Available Topics (Columns)

| Topic | Schema | Class | Messages |
|-------|--------|-------|----------|
| `/instruction` | `Instructions` | **unknown** | 1 |
| `/left-arm-action` | `RobotState` | **action** | 319 |
| `/left-arm-state` | `RobotState` | **state** | 319 |
| `/left-ee-action` | `GripperState` | **action** | 319 |
| `/left-ee-state` | `GripperState` | **state** | 319 |
| `/left-wrist-camera` | `foxglove.CompressedVideo` | **image** | 319 |
| `/right-arm-action` | `RobotState` | **action** | 319 |
| `/right-arm-state` | `RobotState` | **state** | 319 |
| `/right-ee-action` | `GripperState` | **action** | 319 |
| `/right-ee-state` | `GripperState` | **state** | 319 |
| `/right-wrist-camera` | `foxglove.CompressedVideo` | **image** | 319 |
| `/right-wrist-camera-info` | `foxglove.CameraCalibration` | **image** | 1 |
| `/top-camera` | `foxglove.CompressedVideo` | **image** | 319 |
| `/top-camera-info` | `foxglove.CameraCalibration` | **image** | 1 |

## Key Signal Classification

### Image keys (5)
  - `/right-wrist-camera-info`
  - `/top-camera-info`
  - `/top-camera`
  - `/right-wrist-camera`
  - `/left-wrist-camera`

### Action keys (4)
  - `/left-ee-action`
  - `/left-arm-action`
  - `/right-ee-action`
  - `/right-arm-action`

### State keys (4)
  - `/left-ee-state`
  - `/left-arm-state`
  - `/right-ee-state`
  - `/right-arm-state`

### Depth keys (0)
  *(none)*

### Unknown topics (1)
  - `/instruction`

## Integrity Checks
- [x] actions present
- [x] robot state present
- [x] images present
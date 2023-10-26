import { buttonGroup, useControls, button } from 'leva';
import { useContext, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { WebSocketContext } from '../WebSocket/WebSocket';
import { SceneTreeWebSocketListener } from '../Scene/Scene';
import { EventRepeat, WindowOutlined } from '@mui/icons-material';
import store from '../../store';

const msgpack = require('msgpack-lite');

function store_dispatch(path, data) {
  store.dispatch({
    type: 'write',
    path,
    data,
  });
}

function dispatch_and_send(websocket, dispatch, path, data) {
  dispatch({
    type: 'write',
    path,
    data,
  });
  if (websocket.readyState === WebSocket.OPEN) {
    const message = msgpack.encode({
      type: 'write',
      path,
      data,
    });
    websocket.send(message);
  }
}

let last_on_click = null

const onMouseClick = (event) => {
  if (!(event.target.nodeName === "CANVAS")) {
    return;
  }
  const mouseX = event.clientX - event.target.offsetLeft;
  const mouseY = event.clientY - event.target.offsetTop;
  store_dispatch(
    "renderingState/img_x_pos",
    mouseX,
  );
  store_dispatch(
    "renderingState/img_y_pos",
    mouseY,
  )
};

export function RenderControls(props) {
  // connection status indicators
  const sceneTree = props.sceneTree;
  const scene = sceneTree.object;
  const websocket = useContext(WebSocketContext).socket;
  const outputOptions = useSelector(
    (state) => state.renderingState.output_options,
  );
  const outputChoice = useSelector(
    (state) => state.renderingState.output_choice,
  );
  const colormapOptions = useSelector(
    (state) => state.renderingState.colormap_options,
  );
  const colormapChoice = useSelector(
    (state) => state.renderingState.colormap_choice,
  );
  const colormapInvert = useSelector(
    (state) => state.renderingState.colormap_invert,
  );
  const colormapNormalize = useSelector(
    (state) => state.renderingState.colormap_normalize,
  );
  const max_resolution = useSelector(
    (state) => state.renderingState.maxResolution,
  );
  const target_train_util = useSelector(
    (state) => state.renderingState.targetTrainUtil,
  );
  const render_time = useSelector((state) => state.renderingState.renderTime);
  const crop_enabled = useSelector(
    (state) => state.renderingState.crop_enabled,
  );

  const crop_bg_color = useSelector(
    (state) => state.renderingState.crop_bg_color,
  );

  const crop_scale = useSelector((state) => state.renderingState.crop_scale);

  const crop_center = useSelector((state) => state.renderingState.crop_center);

  const dispatch = useDispatch();

  const [display_render_time, set_display_render_time] = useState(true);

  const [enable_sam, set_enable_sam] = useState(true);

  // for sam-nerf
  const mx = useSelector(
    (state) => state.renderingState.img_x_pos,
  );
  const my = useSelector(
    (state) => state.renderingState.img_y_pos,
  );

  // storing selected positions for sam nerf
  const xs = useSelector(
    (state) => state.renderingState.xs,
  );
  const ys = useSelector(
    (state) => state.renderingState.ys,
  );

  const rendering_fps = useSelector(
    (state) => state.renderingState.rendering_fps,
  );
  const num_total_frames = useSelector(
    (state) => state.renderingState.num_total_frames,
  )
  const is_playing = useSelector(
    (state) => state.renderingState.is_playing,
  );
  const fps_first = useSelector(
    (state) => state.renderingState.fps_first,
  );
  const spf = 1. / rendering_fps;
  const stick_to_current_pose = useSelector(
    (state) => state.renderingState.stick_to_current_pose,
  );
  const time_interval = 1. / num_total_frames;
  const [display_play_button, set_display_play_button] = useState(true);

  // const points = useSelector(
  //   (state) => state.renderingState.points,
  // );

  // sam-nerf
  const use_sam = useSelector(
    (state) => state.renderingState.use_sam,
  );

  const receive_temporal_dist = (e) => {
    const msg = msgpack.decode(new Uint8Array(e.data));
    if (msg.path === '/model/has_temporal_distortion') {
      set_display_render_time(msg.data === 'true');
      set_display_play_button(msg.data === 'true');
      websocket.removeEventListener('message', receive_temporal_dist);
    }
  };

  const receive_enable_sam = (e) => {
    const msg = msgpack.decode(new Uint8Array(e.data));
    if (msg.path === '/model/enable_sam') {
      set_enable_sam(msg.data === 'true');
      websocket.removeEventListener('message', receive_enable_sam);
    }
  };

  websocket.addEventListener('message', receive_temporal_dist);
  websocket.addEventListener('message', receive_enable_sam);

  const [, setControls] = useControls(
    () => ({
      // training speed
      SpeedButtonGroup: buttonGroup({
        label: `Train Speed`,
        hint: 'Select the training speed, affects viewer render quality, not final render quality',
        opts: {
          Fast: () =>
            setControls({ target_train_util: 0.9, max_resolution: 512 }),
          Balanced: () =>
            setControls({ target_train_util: 0.7, max_resolution: 1024 }),
          Slow: () =>
            setControls({ target_train_util: 0.1, max_resolution: 2048 }),
        },
      }),
      // output_options
      output_options: {
        label: 'Output Render',
        options: [...new Set(outputOptions)],
        value: outputChoice,
        hint: 'Select the output to render',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/output_choice',
            v,
          );
        },
      },
      // colormap_options
      colormap_options: {
        label: 'Colormap',
        options: colormapOptions,
        value: colormapChoice,
        hint: 'Select the colormap to use',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/colormap_choice',
            v,
          );
        },
        disabled: colormapOptions.length === 1,
      },
      colormap_invert: {
        label: '| Invert',
        value: colormapInvert,
        hint: 'Invert the colormap',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/colormap_invert',
            v,
          );
        },
        render: (get) => get('colormap_options') !== 'default',
      },
      colormap_normalize: {
        label: '| Normalize',
        value: colormapNormalize,
        hint: 'Whether to normalize output between 0 and 1',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/colormap_normalize',
            v,
          );
        },
        render: (get) => get('colormap_options') !== 'default',
      },
      colormap_range: {
        label: '| Range',
        value: [0, 1],
        step: 0.01,
        min: -2,
        max: 5,
        hint: 'Min and max values of the colormap',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/colormap_range',
            v,
          );
        },
        render: (get) => get('colormap_options') !== 'default',
      },
      // Dynamic Resolution
      target_train_util: {
        label: 'Train Util.',
        value: target_train_util,
        min: 0,
        max: 1,
        step: 0.05,
        hint: "Target training utilization, 0.0 is slow, 1.0 is fast, doesn't affect final render quality",
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/targetTrainUtil',
            v,
          );
        },
      },
      // resolution
      max_resolution: {
        label: 'Max Res.',
        value: max_resolution,
        min: 256,
        max: 2048,
        step: 1,
        hint: 'Maximum resolution to render in viewport',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/maxResolution',
            v,
          );
        },
      },
      '  ': buttonGroup({
        '256px': () => setControls({ max_resolution: 256 }),
        '512px': () => setControls({ max_resolution: 512 }),
        '1024px': () => setControls({ max_resolution: 1024 }),
        '2048px': () => setControls({ max_resolution: 2048 }),
      }),
      // Enable Crop
      crop_enabled: {
        label: 'Crop Viewport',
        value: crop_enabled,
        hint: 'Crop the viewport to the selected box',
        onChange: (value) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/crop_enabled',
            value,
          );
        },
      },
      crop_bg_color: {
        label: '| Background Color',
        value: crop_bg_color,
        render: (get) => get('crop_enabled'),
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/crop_bg_color',
            v,
          );
        },
      },
      crop_scale: {
        label: '|  Scale',
        value: crop_scale,
        min: 0,
        max: 10,
        step: 0.05,
        render: (get) => get('crop_enabled'),
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/crop_scale',
            v,
          );
        },
      },
      crop_center: {
        label: '|  Center',
        value: crop_center,
        min: -10,
        max: 10,
        step: 0.05,
        render: (get) => get('crop_enabled'),
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/crop_center',
            v,
          );
        },
      },
      // Dynamic NeRF rendering time
      ...(display_render_time
        ? {
            render_time: {
              label: 'Render Timestep',
              value: render_time,
              min: 0,
              max: 1,
              step: 0.01,
              onChange: (v) => {
                dispatch_and_send(
                  websocket,
                  dispatch,
                  'renderingState/render_time',
                  v,
                );
                dispatch({
                  type: 'write',
                  path: 'renderingState/renderTime',
                  data: v,
                });
              },
            },
          }
        : {}),
      // video nerf play button
      ...(display_play_button
        ? {
          "FPS": {
            value: rendering_fps,
            min: 0,
            max: 30,
            step: 1,
            onChange: (v) => {
              dispatch_and_send(
                websocket,
                dispatch,
                'renderingState/rendering_fps',
                v,
              );
              console.log(rendering_fps);
            },
          },
        } : {}
      ),
      ...(display_play_button
        ? {
          "FPS Fisrt": {
            value: fps_first,
            onChange: (v) => {
              dispatch_and_send(
                websocket,
                dispatch,
                'renderingState/fps_first',
                v,
              );
            },
          },
        } : {}
      ),
      ...(display_play_button
        ? {
          "stick_to_current_pose": {
            label: "Stick to current pose",
            value: stick_to_current_pose,
            onChange: (v) => {
              dispatch_and_send(
                websocket,
                dispatch,
                'renderingState/stick_to_current_pose',
                v,
              );
              // console.log(stick_to_current_pose);
              if (stick_to_current_pose) {
                // sceneTree.metadata.camera_controls.disconnect();
                sceneTree.metadata.camera_controls.enabled = false;
              }
              else {
                // sceneTree.metadata.camera_controls.connect();
                sceneTree.metadata.camera_controls.enabled = true;
              };
            },
          },
        } : {}
      ),
      ...(enable_sam
        ? {
          "Use SAM": {
            label: "Use SAM",
            value: use_sam,
            onChange: (v) => {
              dispatch_and_send(
                websocket,
                dispatch,
                'renderingState/use_sam',
                v,
              );
            },
          },
        } : {}
      ),
      ...(enable_sam
        ? {
          SAMButtonGroup: buttonGroup({
            label: "SAM",
            hint: 'SAM related options',
            opts: {
              "remove all pins": () => {
                dispatch_and_send(
                  websocket,
                  dispatch,
                  'renderingState/xs',
                  [],
                );
                dispatch_and_send(
                  websocket,
                  dispatch,
                  'renderingState/ys',
                  [],
                );
                for (let i = 0; i < window.points.length; i++) {
                  scene.remove(window.points[i]);
                }
              },
              "remove last pin": () => {
                if (xs.length > 0) {
                  dispatch_and_send(
                    websocket,
                    dispatch,
                    'renderingState/xs',
                    xs.slice(0, -1),
                  );
                  dispatch_and_send(
                    websocket,
                    dispatch,
                    'renderingState/ys',
                    ys.slice(0, -1),
                  );
                  scene.remove(window.points.pop());
                }
              }
            }
          }),
        } : {}
      ),
      // "Use SAM": {
      //   label: "Use SAM",
      //   value: use_sam,
      //   onChange: (v) => {
      //     dispatch_and_send(
      //       websocket,
      //       dispatch,
      //       'renderingState/use_sam',
      //       v,
      //     );
      //   },
      // },
      ...(display_play_button
        ? {
          [is_playing ? "pause" : "play"]: button(
            (get) => {
              dispatch_and_send(
                websocket,
                dispatch,
                'renderingState/is_playing',
                !is_playing,
              );
            }
          ),
          }
        : {}),
    }),
    [
      outputOptions,
      outputChoice,
      colormapOptions,
      colormapChoice,
      max_resolution,
      crop_enabled,
      target_train_util,
      render_time,
      display_render_time,
      is_playing,
      display_play_button,
      rendering_fps,
      fps_first,
      stick_to_current_pose,
      // NOTE: if set twice, the onChange func will execute twice
      // use_sam,
      websocket, // need to re-render when websocket changes to use the new websocket
      mx,
      my,
    ],
  );

  useEffect(() => {
    setControls({ max_resolution });
    setControls({ output_options: outputChoice });
    setControls({ colormap_options: colormapChoice });
    setControls({ crop_enabled });
    setControls({ crop_bg_color });
    setControls({ crop_scale });
    setControls({ crop_center });
    // setControls({ stick_to_current_pose });
    let intervalId = null;
    if (is_playing) {
      // console.log("time interval: " + time_interval);
      intervalId = setInterval(() => {
        setControls({ render_time: render_time + time_interval });
      },
        spf * 1000,
      );
    }
    return () => {
      if (intervalId !== null) {
        clearInterval(intervalId);
      }
    };
  }, [
    setControls,
    outputOptions,
    outputChoice,
    colormapOptions,
    colormapChoice,
    max_resolution,
    target_train_util,
    render_time,
    crop_enabled,
    crop_bg_color,
    crop_scale,
    crop_center,
    display_render_time,
    display_play_button,
    rendering_fps,
    is_playing,
    fps_first,
    stick_to_current_pose,
    use_sam,
  ]);

  return null;
}

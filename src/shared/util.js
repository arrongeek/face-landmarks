/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as tf from '@tensorflow/tfjs-core';

import {GREEN, LABEL_TO_COLOR, NUM_IRIS_KEYPOINTS, NUM_KEYPOINTS, RED, TUNABLE_FLAG_VALUE_RANGE_MAP} from './params';
import {TRIANGULATION} from './triangulation';
import * as THREE from 'three';
import { ARButton } from 'three/examples/jsm/webxr/ARButton';
export function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

export function isMobile() {
  return isAndroid() || isiOS();
}

const img = new Image();
img.src = 'https://s3.ap-northeast-2.amazonaws.com/test-admin.geekstudio.kr/png-transparent-black-framed-aviator-style-sunglasses-illustration-aviator-sunglasses-sunglasses-lens-copyright-glasses-removebg-preview.png';

/**
 * Reset the target backend.
 *
 * @param backendName The name of the backend to be reset.
 */
async function resetBackend(backendName) {
  const ENGINE = tf.engine();
  if (!(backendName in ENGINE.registryFactory)) {
    throw new Error(`${backendName} backend is not registered.`);
  }

  if (backendName in ENGINE.registry) {
    const backendFactory = tf.findBackendFactory(backendName);
    tf.removeBackend(backendName);
    tf.registerBackend(backendName, backendFactory);
  }

  await tf.setBackend(backendName);
}

/**
 * Set environment flags.
 *
 * This is a wrapper function of `tf.env().setFlags()` to constrain users to
 * only set tunable flags (the keys of `TUNABLE_FLAG_TYPE_MAP`).
 *
 * ```js
 * const flagConfig = {
 *        WEBGL_PACK: false,
 *      };
 * await setEnvFlags(flagConfig);
 *
 * console.log(tf.env().getBool('WEBGL_PACK')); // false
 * console.log(tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')); // false
 * ```
 *
 * @param flagConfig An object to store flag-value pairs.
 */
export async function setBackendAndEnvFlags(flagConfig, backend) {
  if (flagConfig == null) {
    return;
  } else if (typeof flagConfig !== 'object') {
    throw new Error(
        `An object is expected, while a(n) ${typeof flagConfig} is found.`);
  }

  // Check the validation of flags and values.
  for (const flag in flagConfig) {
    // TODO: check whether flag can be set as flagConfig[flag].
    if (!(flag in TUNABLE_FLAG_VALUE_RANGE_MAP)) {
      throw new Error(`${flag} is not a tunable or valid environment flag.`);
    }
    if (TUNABLE_FLAG_VALUE_RANGE_MAP[flag].indexOf(flagConfig[flag]) === -1) {
      throw new Error(
          `${flag} value is expected to be in the range [${
              TUNABLE_FLAG_VALUE_RANGE_MAP[flag]}], while ${flagConfig[flag]}` +
          ' is found.');
    }
  }

  tf.env().setFlags(flagConfig);

  const [runtime, $backend] = backend.split('-');

  if (runtime === 'tfjs') {
    await resetBackend($backend);
  }
}

function distance(a, b) {
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

/**
 * @params ctx 부위 스타일
 * @params points 부위 좌표
*/
function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  // moveTo(x, y) 펜의 위치를 새로운 좌표 (x, y)로 이동시킵니다. 선은 그리지 않고 이동만 합니다.
  // region.moveTo(points[0][0], points[0][1]);
  // for (let i = 1; i < points.length; i++) {
  //   const point = points[i];
  //   // lineTo(x, y) 현재위치에서 (x, y)까지 선을 그립니다.
  //   region.lineTo(point[0], point[1]);
  // }
  //
  // if (closePath) {
  //   region.closePath();
  // }
  ctx.beginPath(); // 경로를 다시 재설정해줌
  ctx.moveTo(points[0][0], points[0][1]);
  const x1 = points[0][0], y1 = points[0][1];
  // for (let i = 1; i < points.length; i++) {
    // const point = points[i];
    // // lineTo(x, y) 현재위치에서 (x, y)까지 선을 그립니다.
    // console.log(point)
    //
    // const x2 = point[0], y2 = point[1];
    // const cpx = (x1 + x2) / 2;
    // const cpy = (y1 + y2) / 2 - 50;
    // // ctx.quadraticCurveTo(x2, y2, x2, y2); // 위쪽 곡선
    // // ctx.quadraticCurveTo(y2, x2, x2, y2); // 아래쪽 곡선
    // ctx.lineTo(point[0], point[1])
  // }

  for (let i = 1; i < points.length - 2; i++) {
    const [x1, y1] = points[i - 1];
    const [x2, y2] = points[i];
    const cpx = (x1 + x2) / 2;
    const cpy = (y1 + y2) / 2;

    // 곡선 형태 연결
    ctx.quadraticCurveTo(x1, y1, cpx, cpy);
  }

  ctx.closePath();
  ctx.fillStyle = "rgba(230, 92, 72, 0.2)";
  ctx.fill('evenodd');


  // 스파클링 효과
  // sparkleAlongPath(ctx, points);

  // ctx.stroke(region);
  // 그라이데이션
  // const gradient = ctx.createLinearGradient(50, 100, 150, 100);
  // gradient.addColorStop(0, "red");
  // gradient.addColorStop(1, "darkred");
  //
  // ctx.fillStyle = gradient;
  // ctx.fill();
}

// 스파클링 효과
function sparkleAlongPath(ctx, points) {
  for (let i = 0; i < points.length; i++) {
    const x = points[i][0];
    const y = points[i][1];
    const size = Math.random() + 0.005; // 반짝이는 스파클링 사이즈
    const opacity = Math.random(); //

    // 경로의 각 점에 반짝이 원을 그리기
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(238, 130, 238, ${opacity})`;
    ctx.fill();
  }
}

/**
 * Draw the keypoints on the video.
 * @param ctx 2D rendering context.
 * @param faces A list of faces to render.
 * @param triangulateMesh Whether or not to display the triangle mesh.
 * @param boundingBox Whether or not to display the bounding box.
 */
export function drawResults(ctx, faces, triangulateMesh, boundingBox) {
  faces.forEach((face) => {
    const keypoints =
        face.keypoints.map((keypoint) => [keypoint.x, keypoint.y]);

    // 얼굴 인식 네모박스 주석처리
    // if (boundingBox) {
    //   ctx.strokeStyle = RED;
    //   ctx.lineWidth = 1;
    //
    //   const box = face.box;
    //   drawPath(
    //       ctx,
    //       [
    //         [box.xMin, box.yMin], [box.xMax, box.yMin], [box.xMax, box.yMax],
    //         [box.xMin, box.yMax]
    //       ],
    //       true);
    // }
    //

    // 얼굴 형태 구조
    // if (triangulateMesh) {
    //   ctx.strokeStyle = GREEN;
    //   ctx.lineWidth = 0.5;
    //
    //   for (let i = 0; i < TRIANGULATION.length / 3; i++) {
    //     const points = [
    //       TRIANGULATION[i * 3],
    //       TRIANGULATION[i * 3 + 1],
    //       TRIANGULATION[i * 3 + 2],
    //     ].map((index) => keypoints[index]);
    //
    //     drawPath(ctx, points, true);
    //   }
    // } else {
    //   ctx.fillStyle = GREEN;
    //
    //   for (let i = 0; i < NUM_KEYPOINTS; i++) {
    //     const x = keypoints[i][0];
    //     const y = keypoints[i][1];
    //
    //     ctx.beginPath();
    //     ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
    //     ctx.fill();
    //   }
    // }

    // 얼굴 랜드마크 다듬는 기능
    /**
    if (keypoints.length > NUM_KEYPOINTS) {
      // 눈 동공 색상
      ctx.strokeStyle = RED;
      ctx.lineWidth = 1;

      // 왼쪽 동공
      const leftCenter = keypoints[NUM_KEYPOINTS];
      const leftDiameterY =
          distance(keypoints[NUM_KEYPOINTS + 4], keypoints[NUM_KEYPOINTS + 2]);
      const leftDiameterX =
          distance(keypoints[NUM_KEYPOINTS + 3], keypoints[NUM_KEYPOINTS + 1]);

      ctx.beginPath();
      // ellips(x, y, radiusX, radiusY, rotation, startAngle, endAngle, counterclockwise)
      // 중심이 (x, y)이고 radiusX와 radiusY를 가진 타원을 그립니다.
      ctx.ellipse(
          leftCenter[0], leftCenter[1], leftDiameterX / 2, leftDiameterY / 2, 0,
          0, 2 * Math.PI);
      ctx.stroke();

      // 오른쪽 동공
      if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
        const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
        const rightDiameterY = distance(
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
        const rightDiameterX = distance(
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]);

        ctx.beginPath();
        ctx.ellipse(
            rightCenter[0], rightCenter[1], rightDiameterX / 2,
            rightDiameterY / 2, 0, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
      */

    const contours = faceLandmarksDetection.util.getKeypointIndexByContour(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh);

    // 왼쪽 오른쪽 눈썹 외 부위 제거
    // lips는 for of 문 안에서 faceOval 처럼 제외처리가 안됨
    // 눈썹: leftEyebrow / rightEyebrow
    // 눈동자: leftIris / rightIris
    // 입술: lips
    // 얼굴면적 전체: faceOval
    // const eyeBrow = Object.entr/**/ies(contours).filter(data => data[0] === 'leftEyebrow' || data[0] === 'rightEyebrow');
    const lips = Object.entries(contours).filter(data => data[0] === 'lips');

    for (const [label, contour] of lips) {
      ctx.lineWidth = 3;

      // 실시간 입의 움직임 좌표(?) ex: [273.1239218412848, 265.23125125215]
      const path = contour.map((index) => keypoints[index]);
      if (path.every(value => value != undefined)) {
        drawPath(ctx, path, true);
      }
    }
  });
}

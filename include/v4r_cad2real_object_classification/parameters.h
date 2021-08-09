#pragma once

struct TransfoParams {
  // PC transformations
  float to_remove;
  unsigned int to_keep;
  float occl_pct;
  float noise_std;
  unsigned int rotation_deg;
};

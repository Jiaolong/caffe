#ifndef CAFFE_DUAL_SOURCE_LAYER_HPP_
#define CAFFE_DUAL_SOURCE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/blocking_queue.hpp"

#define NUM_JOINTS 14

namespace caffe {

/**
 * @brief Provides dual source data to the Net from windows
 * of images files, specified by a window data file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class PoseWindowDataLayer : 
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit PoseWindowDataLayer(const LayerParameter& param);
  virtual ~PoseWindowDataLayer();
  
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseWindowData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;
 
 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch);

  virtual unsigned int PrefetchRand();

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);

  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch<Dtype>* > prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum TopField { LOCAL_DATA, JOINT_LABEL, POSE_2D, TNUM };
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
  // all joint positions of the dataset, each element stores
  // joints of one image
  vector<std::map<int, std::vector<Dtype> > > image_joint_2d_pos_;

  int num_joints_;
  int num_labels_;
};

}  // namespace caffe

#endif  // CAFFE_DUAL_SOURCE_LAYER_HPP_

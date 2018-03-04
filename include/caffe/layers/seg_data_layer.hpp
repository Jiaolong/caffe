#ifndef CAFFE_SEG_DATA_LAYER_HPP_
#define CAFFE_SEG_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class SegDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit SegDataLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~SegDataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SegData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
	shared_ptr<Caffe::RNG> prefetch_rng_;
	virtual void ShuffleImages();
    virtual void load_batch(Batch<Dtype>* batch);

#ifdef USE_MPI
	inline virtual void advance_cursor(){
		lines_id_+= batch_size_;
		if (lines_id_ >= lines_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ -= lines_.size();
			if (this->layer_param_.seg_data_param().shuffle()) {
				ShuffleImages();
			}
		}
	}
#endif

	vector<std::pair<std::string, std::string> > lines_;
	int lines_id_;
	int batch_size_;
	string name_pattern_;
};
}  // namespace caffe

#endif  // CAFFE_SEG_DATA_LAYER_HPP_

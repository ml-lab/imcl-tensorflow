/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_

#include <functional>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

// Status callback.
typedef std::function<void(const Status&)> StatusCallback;

// Allocator callback for out-of-band transfers.
class TensorShape;
typedef std::function<void*(size_t, const DataType&, const TensorShape&)>
    TensorBufAllocator;

// Interface for talking with the TensorFlow Worker service.
class WorkerInterface {
 public:
  virtual ~WorkerInterface() {}

  virtual void GetStatusAsync(const GetStatusRequest* request,
                              GetStatusResponse* response,
                              StatusCallback done) = 0;

  virtual void RegisterGraphAsync(const RegisterGraphRequest* request,
                                  RegisterGraphResponse* response,
                                  StatusCallback done) = 0;

  virtual void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                    DeregisterGraphResponse* response,
                                    StatusCallback done) = 0;

  virtual void RunGraphAsync(CallOptions* opts, const RunGraphRequest* request,
                             RunGraphResponse* response,
                             StatusCallback done) = 0;

  virtual void CleanupGraphAsync(const CleanupGraphRequest* request,
                                 CleanupGraphResponse* response,
                                 StatusCallback done) = 0;

  virtual void CleanupAllAsync(const CleanupAllRequest* request,
                               CleanupAllResponse* response,
                               StatusCallback done) = 0;

  virtual void RecvTensorAsync(CallOptions* opts,
                               const RecvTensorRequest* request,
                               RecvTensorResponse* response,
                               TensorBufAllocator allocator,
                               StatusCallback done) = 0;

  virtual void LoggingAsync(const LoggingRequest* request,
                            LoggingResponse* response, StatusCallback done) = 0;

  virtual void TracingAsync(const TracingRequest* request,
                            TracingResponse* response, StatusCallback done) = 0;

  Status GetStatus(const GetStatusRequest* request,
                   GetStatusResponse* response) {
    return CallAndWait(&ME::GetStatusAsync, request, response);
  }

  Status RegisterGraph(const RegisterGraphRequest* request,
                       RegisterGraphResponse* response) {
    return CallAndWait(&ME::RegisterGraphAsync, request, response);
  }

  Status DeregisterGraph(const DeregisterGraphRequest* request,
                         DeregisterGraphResponse* response) {
    return CallAndWait(&ME::DeregisterGraphAsync, request, response);
  }

  Status CleanupGraph(const CleanupGraphRequest* request,
                      CleanupGraphResponse* response) {
    return CallAndWait(&ME::CleanupGraphAsync, request, response);
  }

  Status CleanupAll(const CleanupAllRequest* request,
                    CleanupAllResponse* response) {
    return CallAndWait(&ME::CleanupAllAsync, request, response);
  }

  Status Logging(const LoggingRequest* request, LoggingResponse* response) {
    return CallAndWait(&ME::LoggingAsync, request, response);
  }

  Status Tracing(const TracingRequest* request, TracingResponse* response) {
    return CallAndWait(&ME::TracingAsync, request, response);
  }

 private:
  typedef WorkerInterface ME;

  template <typename Method, typename Req, typename Resp>
  Status CallAndWait(Method func, const Req* req, Resp* resp) {
    Status ret;
    Notification n;
    (this->*func)(req, resp, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_

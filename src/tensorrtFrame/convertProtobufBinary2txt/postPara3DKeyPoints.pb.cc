// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: postPara3DKeyPoints.proto

#include "postPara3DKeyPoints.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
class C3DParaDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<C3DPara> _instance;
} _C3DPara_default_instance_;
static void InitDefaultsscc_info_C3DPara_postPara3DKeyPoints_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::_C3DPara_default_instance_;
    new (ptr) ::C3DPara();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::C3DPara::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_C3DPara_postPara3DKeyPoints_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_C3DPara_postPara3DKeyPoints_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_postPara3DKeyPoints_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_postPara3DKeyPoints_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_postPara3DKeyPoints_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_postPara3DKeyPoints_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::C3DPara, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::C3DPara, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::C3DPara, wshpbase_),
  PROTOBUF_FIELD_OFFSET(::C3DPara, wexpbase_),
  ~0u,
  ~0u,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::C3DPara)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::_C3DPara_default_instance_),
};

const char descriptor_table_protodef_postPara3DKeyPoints_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\031postPara3DKeyPoints.proto\"-\n\007C3DPara\022\020"
  "\n\010wShpBase\030\001 \003(\002\022\020\n\010wExpBase\030\002 \003(\002"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_postPara3DKeyPoints_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_postPara3DKeyPoints_2eproto_sccs[1] = {
  &scc_info_C3DPara_postPara3DKeyPoints_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_postPara3DKeyPoints_2eproto_once;
static bool descriptor_table_postPara3DKeyPoints_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_postPara3DKeyPoints_2eproto = {
  &descriptor_table_postPara3DKeyPoints_2eproto_initialized, descriptor_table_protodef_postPara3DKeyPoints_2eproto, "postPara3DKeyPoints.proto", 74,
  &descriptor_table_postPara3DKeyPoints_2eproto_once, descriptor_table_postPara3DKeyPoints_2eproto_sccs, descriptor_table_postPara3DKeyPoints_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_postPara3DKeyPoints_2eproto::offsets,
  file_level_metadata_postPara3DKeyPoints_2eproto, 1, file_level_enum_descriptors_postPara3DKeyPoints_2eproto, file_level_service_descriptors_postPara3DKeyPoints_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_postPara3DKeyPoints_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_postPara3DKeyPoints_2eproto)), true);

// ===================================================================

void C3DPara::InitAsDefaultInstance() {
}
class C3DPara::_Internal {
 public:
  using HasBits = decltype(std::declval<C3DPara>()._has_bits_);
};

C3DPara::C3DPara()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:C3DPara)
}
C3DPara::C3DPara(const C3DPara& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr),
      _has_bits_(from._has_bits_),
      wshpbase_(from.wshpbase_),
      wexpbase_(from.wexpbase_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:C3DPara)
}

void C3DPara::SharedCtor() {
}

C3DPara::~C3DPara() {
  // @@protoc_insertion_point(destructor:C3DPara)
  SharedDtor();
}

void C3DPara::SharedDtor() {
}

void C3DPara::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const C3DPara& C3DPara::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_C3DPara_postPara3DKeyPoints_2eproto.base);
  return *internal_default_instance();
}


void C3DPara::Clear() {
// @@protoc_insertion_point(message_clear_start:C3DPara)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  wshpbase_.Clear();
  wexpbase_.Clear();
  _has_bits_.Clear();
  _internal_metadata_.Clear();
}

const char* C3DPara::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // repeated float wShpBase = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 13)) {
          ptr -= 1;
          do {
            ptr += 1;
            _internal_add_wshpbase(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr));
            ptr += sizeof(float);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<13>(ptr));
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedFloatParser(_internal_mutable_wshpbase(), ptr, ctx);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated float wExpBase = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 21)) {
          ptr -= 1;
          do {
            ptr += 1;
            _internal_add_wexpbase(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr));
            ptr += sizeof(float);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<21>(ptr));
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedFloatParser(_internal_mutable_wexpbase(), ptr, ctx);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* C3DPara::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:C3DPara)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated float wShpBase = 1;
  for (int i = 0, n = this->_internal_wshpbase_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(1, this->_internal_wshpbase(i), target);
  }

  // repeated float wExpBase = 2;
  for (int i = 0, n = this->_internal_wexpbase_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(2, this->_internal_wexpbase(i), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:C3DPara)
  return target;
}

size_t C3DPara::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:C3DPara)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated float wShpBase = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->_internal_wshpbase_size());
    size_t data_size = 4UL * count;
    total_size += 1 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_wshpbase_size());
    total_size += data_size;
  }

  // repeated float wExpBase = 2;
  {
    unsigned int count = static_cast<unsigned int>(this->_internal_wexpbase_size());
    size_t data_size = 4UL * count;
    total_size += 1 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_wexpbase_size());
    total_size += data_size;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void C3DPara::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:C3DPara)
  GOOGLE_DCHECK_NE(&from, this);
  const C3DPara* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<C3DPara>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:C3DPara)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:C3DPara)
    MergeFrom(*source);
  }
}

void C3DPara::MergeFrom(const C3DPara& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:C3DPara)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  wshpbase_.MergeFrom(from.wshpbase_);
  wexpbase_.MergeFrom(from.wexpbase_);
}

void C3DPara::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:C3DPara)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void C3DPara::CopyFrom(const C3DPara& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:C3DPara)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool C3DPara::IsInitialized() const {
  return true;
}

void C3DPara::InternalSwap(C3DPara* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  wshpbase_.InternalSwap(&other->wshpbase_);
  wexpbase_.InternalSwap(&other->wexpbase_);
}

::PROTOBUF_NAMESPACE_ID::Metadata C3DPara::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::C3DPara* Arena::CreateMaybeMessage< ::C3DPara >(Arena* arena) {
  return Arena::CreateInternal< ::C3DPara >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
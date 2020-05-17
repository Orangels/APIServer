// namespace DBS 
// {
class CNonCopyable
{
public:
	CNonCopyable(){}
	~CNonCopyable(){}
private: 
	CNonCopyable( const CNonCopyable& );
	CNonCopyable& operator=( const CNonCopyable& );
};
// }
